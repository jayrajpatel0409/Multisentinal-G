import re, sys
def inline(path):
    s = open(path).read()
    def rep(m):
        f = m.group(1); f = f if f.endswith(".tex") else f + ".tex"
        return inline(f).rstrip("\n")
    return re.sub(r"^\\input\{([^}]+)\}[ \t]*$", rep, s, flags=re.M)
out = inline("main.tex")
# tables fragments are inlined at line level; also handle indented \input inside tabular
open("MultiSentinel_G_manuscript.tex", "w").write(out)
