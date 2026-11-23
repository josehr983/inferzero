// LFM2 / LFM2.5 (Liquid AI): the output parser of its chat format. The prompt comes from the GGUF's
// template (chat_jinja.h); the reply is "<think>reasoning</think>answer", a tool round ends with
// <|tool_call_start|>[name(arg='text', n=3, opts={"a": 1})]<|tool_call_end|>: Python call syntax, strings as Python
// literals, lists and objects as JSON (the template's format_arg_value).
#pragma once
#include "chat.h"

namespace chat {

struct LfmPy {
  const std::string& s;
  size_t i = 0;
  bool ok = true;
  void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i; }
  // a Python / JSON string literal starting at s[i] (a quote)
  std::string str() {
    const char q = s[i++];
    std::string out;
    while (i < s.size() && s[i] != q) {
      if (s[i] == '\\' && i + 1 < s.size()) {
        const char c = s[i + 1];
        i += 2;
        if (c == 'n') out += '\n';
        else if (c == 'r') out += '\r';
        else if (c == 't') out += '\t';
        else if (c == 'u' && i + 4 <= s.size()) {
          const unsigned cp = (unsigned)strtoul(s.substr(i, 4).c_str(), nullptr, 16);
          i += 4;
          if (cp < 0x80) out += (char)cp;
          else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 63)); }
          else { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 63)); out += (char)(0x80 | (cp & 63)); }
        } else out += c;  // (\\ \' \" and anything else: the character itself)
        continue;
      }
      out += s[i++];
    }
    if (i < s.size()) ++i;  // the closing quote
    else ok = false;
    return out;
  }
  Json value(int depth = 0) {
    ws();
    if (i >= s.size() || depth > 64) { ok = false; return Json(); }
    const char c = s[i];
    if (c == '\'' || c == '"') return Json(str());
    if (c == '[' || c == '(') {
      const char close = c == '[' ? ']' : ')';
      ++i;
      Json a = Json::array();
      for (;;) {
        ws();
        if (i >= s.size()) { ok = false; break; }
        if (s[i] == close) { ++i; break; }
        if (s[i] == ',') { ++i; continue; }
        a.push(value(depth + 1));
        if (!ok) break;
      }
      return a;
    }
    if (c == '{') {
      ++i;
      Json o = Json::object();
      for (;;) {
        ws();
        if (i >= s.size()) { ok = false; break; }
        if (s[i] == '}') { ++i; break; }
        if (s[i] == ',') { ++i; continue; }
        const Json k = value(depth + 1);
        ws();
        if (!ok || i >= s.size() || s[i] != ':') { ok = false; break; }
        ++i;
        o.set(k.is_string() ? k.as_string() : k.dump(), value(depth + 1));
        if (!ok) break;
      }
      return o;
    }
    // a bare word: a number, True / False / None (or their JSON forms), else text up to the next separator
    size_t e = i;
    while (e < s.size() && s[e] != ',' && s[e] != ')' && s[e] != ']' && s[e] != '}' && s[e] != '\n') ++e;
    std::string w = s.substr(i, e - i);
    while (!w.empty() && w.back() == ' ') w.pop_back();
    i = e;
    if (w == "True" || w == "true") return Json(true);
    if (w == "False" || w == "false") return Json(false);
    if (w == "None" || w == "null") return Json();
    Json num;
    if (!w.empty() && (isdigit((unsigned char)w[0]) || w[0] == '-' || w[0] == '+' || w[0] == '.') && Json::try_parse(w, num) && num.is_number())
      return num;
    return Json(w);
  }
};

// The calls of text: every <|tool_call_start|> ... <|tool_call_end|> block (the last may be unclosed at the end).
inline std::vector<ToolCall> parse_lfm_tool_calls(const std::string& text) {
  static const std::string open = "<|tool_call_start|>", close = "<|tool_call_end|>";
  std::vector<ToolCall> calls;
  size_t pos = 0;
  while ((pos = text.find(open, pos)) != std::string::npos) {
    const size_t b = pos + open.size(), e = text.find(close, b);
    const std::string body = text.substr(b, e == std::string::npos ? std::string::npos : e - b);
    pos = e == std::string::npos ? text.size() : e + close.size();
    LfmPy p{body};
    p.ws();
    if (p.i < body.size() && body[p.i] == '[') ++p.i;
    for (;;) {
      p.ws();
      while (p.i < body.size() && (body[p.i] == ',' || body[p.i] == ' ' || body[p.i] == '\n')) ++p.i;
      if (p.i >= body.size() || body[p.i] == ']') break;
      size_t n0 = p.i;
      while (p.i < body.size() && (isalnum((unsigned char)body[p.i]) || body[p.i] == '_' || body[p.i] == '.' || body[p.i] == '-')) ++p.i;
      ToolCall tc;
      tc.name = body.substr(n0, p.i - n0);
      p.ws();
      if (tc.name.empty() || p.i >= body.size() || body[p.i] != '(') break;
      ++p.i;
      int positional = 0;
      for (;;) {
        p.ws();
        if (p.i >= body.size()) break;
        if (body[p.i] == ')') { ++p.i; break; }
        if (body[p.i] == ',') { ++p.i; continue; }
        // key = value (a value without a key: arg0, arg1, ...)
        size_t k0 = p.i, k1 = p.i;
        while (k1 < body.size() && (isalnum((unsigned char)body[k1]) || body[k1] == '_')) ++k1;
        size_t eq = k1;
        while (eq < body.size() && body[eq] == ' ') ++eq;
        std::string key;
        if (k1 > k0 && eq < body.size() && body[eq] == '=' && !(eq + 1 < body.size() && body[eq + 1] == '=')) {
          key = body.substr(k0, k1 - k0);
          p.i = eq + 1;
        } else {
          key = "arg" + std::to_string(positional++);
        }
        tc.arguments.set(key, p.value());
        if (!p.ok) break;
      }
      calls.push_back(std::move(tc));
      if (!p.ok) break;
    }
  }
  return calls;
}

// Splits generated text into reasoning, visible content and tool calls, incrementally for streaming.
class LfmOutputParser : public OutputParserBase {
 public:
  LfmOutputParser(bool thinking, const Json& tools) : in_reasoning_(thinking), tools_(tools) {}

  void update(const std::string& text, bool final, std::string& d_reason, std::string& d_content) override {
    d_reason.clear();
    d_content.clear();
    if (in_reasoning_) {
      const size_t e = text.find("</think>");
      if (e == std::string::npos) {
        const size_t safe = final ? text.size() : hold_back(text, text.size(), "</think>");
        emit(text, reason_end_, std::max(reason_end_, safe), d_reason);
        return;
      }
      emit(text, reason_end_, e, d_reason);
      in_reasoning_ = false;
      content_start_ = e + 8;
      while (content_start_ < text.size() && text[content_start_] == '\n') ++content_start_;
      content_end_ = content_start_;
      if (content_start_ >= text.size() && !final) return;
    }
    if (content_start_ == std::string::npos) content_start_ = content_end_ = 0;
    if (tool_started_) return;
    if (content_end_ == content_start_) {
      while (content_end_ < text.size() && (text[content_end_] == '\n' || text[content_end_] == ' ' || text[content_end_] == '\r'))
        ++content_end_;
      content_start_ = content_end_;
      if (content_end_ >= text.size()) return;
    }
    size_t limit = text.size();
    const size_t tc = text.find(kOpen, content_start_);
    if (tc != std::string::npos) {
      tool_started_ = true;
      limit = tc;
      while (limit > content_end_ && (text[limit - 1] == '\n' || text[limit - 1] == ' ')) --limit;
    } else if (!final) {
      limit = hold_back(text, text.size(), kOpen);
    }
    if (limit > content_end_) emit(text, content_end_, limit, d_content);
  }

  std::string reasoning_text(const std::string& text) const override {
    const size_t e = text.find("</think>");
    return in_reasoning_start_ ? trim(text.substr(0, e)) : "";
  }
  std::string content_text(const std::string& text) const override {
    size_t s = 0;
    if (in_reasoning_start_) {
      const size_t e = text.find("</think>");
      if (e == std::string::npos) return "";
      s = e + 8;
    }
    std::string c = text.substr(s);
    const size_t tc = c.find(kOpen);
    if (tc != std::string::npos) c = c.substr(0, tc);
    return trim(c);
  }
  std::vector<ToolCall> tool_calls(const std::string& text) const override {
    const size_t e = in_reasoning_start_ ? text.find("</think>") : 0;
    if (e == std::string::npos) return {};
    return parse_lfm_tool_calls(text.substr(e));
  }
  // (a call is given whole once its block has closed or the reply has ended: its arguments never shrink)
  std::vector<PartialToolCall> partial_tool_calls(const std::string& text, bool final) const override {
    const size_t e = in_reasoning_start_ ? text.find("</think>") : 0;
    if (e == std::string::npos) return {};
    std::string t = text.substr(e);
    if (!final) {
      const size_t last = t.rfind("<|tool_call_end|>");
      if (last == std::string::npos) return {};
      t.resize(last + 17);
    }
    std::vector<PartialToolCall> out;
    for (const ToolCall& c : parse_lfm_tool_calls(t)) out.push_back({c.name, c.arguments.dump(), true});
    return out;
  }

 private:
  static constexpr const char* kOpen = "<|tool_call_start|>";
  bool in_reasoning_, in_reasoning_start_ = in_reasoning_;
  Json tools_;
  size_t reason_end_ = 0, content_start_ = std::string::npos, content_end_ = 0;
  bool tool_started_ = false;

  static size_t hold_back(const std::string& text, size_t end, const std::string& tag) {
    for (size_t k = std::min(tag.size() - 1, end); k > 0; --k)
      if (text.compare(end - k, k, tag, 0, k) == 0) return end - k;
    return end;
  }
  void emit(const std::string& text, size_t& from, size_t to, std::string& out) {
    if (to <= from) return;
    const size_t t = utf8_end(text, from, to);
    out = text.substr(from, t - from);
    if (&from == &reason_end_ && from == 0) {
      size_t k = 0;
      while (k < out.size() && out[k] == '\n') ++k;
      out = out.substr(k);
    }
    from = t;
  }
};

}  // namespace chat
