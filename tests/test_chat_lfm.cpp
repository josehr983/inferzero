// CPU-only tests of LFM2's reply parser (src/chat_lfm.h): its Python-style tool calls, malformed ones included (they
// must end, with what was read so far), and the streaming split. Build: .\build.ps1 tests\test_chat_lfm.cpp
#include "../src/chat_lfm.h"
#include <cstdio>

static int fails = 0;
static void expect(bool ok, const char* what) {
  printf("%s %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++fails;
}

int main() {
  using chat::parse_lfm_tool_calls;
  const std::string o = "<|tool_call_start|>", c = "<|tool_call_end|>";
  {
    const auto v = parse_lfm_tool_calls(o + "[get_weather(city='La Paz', unit=\"c\", days=3, deep=True, x=None)]" + c);
    expect(v.size() == 1 && v[0].name == "get_weather", "one call: its name");
    expect(v.size() == 1 && v[0].arguments.dump() == "{\"city\":\"La Paz\",\"unit\":\"c\",\"days\":3,\"deep\":true,\"x\":null}",
           "one call: strings, a number, True, None");
  }
  {
    const auto v = parse_lfm_tool_calls(o + "[a(p='C:\\\\tmp\\n', q='it\\'s'), b(l=[1, 'two', {\"k\": [true, null]}], t=(1, 2))]" + c);
    expect(v.size() == 2 && v[0].name == "a" && v[1].name == "b", "two calls");
    expect(v.size() == 2 && v[0].arguments.dump() == "{\"p\":\"C:\\\\tmp\\n\",\"q\":\"it's\"}", "escapes in Python strings");
    expect(v.size() == 2 && v[1].arguments.dump() == "{\"l\":[1,\"two\",{\"k\":[true,null]}],\"t\":[1,2]}", "lists, objects, a tuple");
  }
  {
    const auto v = parse_lfm_tool_calls(o + "[f()]" + c + " text " + o + "[g(1, 'x')]" + c);
    expect(v.size() == 2 && v[0].arguments.dump() == "{}" && v[1].arguments.dump() == "{\"arg0\":1,\"arg1\":\"x\"}",
           "no arguments; two blocks; positional arguments");
  }
  // malformed: each must return (these did not, before the bare-word check: the argument and array loops span)
  for (const char* bad : {"[f(a=1]", "[f(x=(1, 2])]", "[f(a=)]", "[f(a=[1,,}])]", "[f(a={'k': })]", "[f(a='open", "[f(a={\"k\" 1})]",
                          "[f(", "[", "", "[f(a=1) g(b=2)]", "[(a=1)]", "[f(a=1)))]", "f(a=1"}) {
    const auto v = parse_lfm_tool_calls(o + bad + c);
    char what[96];
    snprintf(what, sizeof what, "malformed ends: %s (%zu call%s)", bad, v.size(), v.size() == 1 ? "" : "s");
    expect(v.size() <= 2, what);
  }
  expect(parse_lfm_tool_calls(o + "[f(a=1]").size() == 1, "malformed and unclosed: the call read so far");
  {
    chat::LfmOutputParser p(true, Json());
    const std::string full = "think it over</think>\n\nHello" + o + "[f(a='x')]" + c;
    std::string dr, dc, reason, content;
    for (size_t n = 1; n <= full.size(); ++n) {
      p.update(full.substr(0, n), n == full.size(), dr, dc);
      reason += dr;
      content += dc;
      if (n < full.size() - 1 && !p.partial_tool_calls(full.substr(0, n), false).empty())
        expect(false, "a call given before its block closed");
    }
    expect(reason == "think it over", "streaming: the reasoning");
    expect(content == "Hello", "streaming: the content, without the call's marker");
    const auto pc = p.partial_tool_calls(full, true);
    expect(pc.size() == 1 && pc[0].name == "f" && pc[0].args == "{\"a\":\"x\"}" && pc[0].done, "streaming: the call, whole");
    expect(p.tool_calls(full).size() == 1 && p.content_text(full) == "Hello" && p.reasoning_text(full) == "think it over", "the final split");
  }
  printf("%s\n", fails ? "FAILED" : "all passed");
  return fails ? 1 : 0;
}
