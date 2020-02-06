import execjs
# 需要安装 PyExecJs
# pip3 install PyExecJs

# js 环境
print(execjs.get().name)

with open('./test.js') as f:
  ctx = execjs.compile(f.read())

  # 获取抽象语法树对象
  ast = ctx.eval('tree')
  print(ast)

  # 调用 latex 解释器
  ast2 = ctx.call("parse", "\\sqrt { \\sin ( \\frac { \\Pi } { 2 } ) } = 1")
  print(ast2)
  print(type(ast2[0]))

  # MathML 的 AST
  exp = ctx.eval("exp")
  print(exp)

  # MathML 字符串
  s = ctx.eval("s")
  print(s)
  print(type(s))

  s2 = ctx.call("renderTreeToString2", ast2, "\\sqrt { \\sin ( \\frac { \\Pi } { 2 } ) } = 1")
  print(s2)
  print(type(s2))
