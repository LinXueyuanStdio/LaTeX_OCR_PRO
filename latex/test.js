var katex = require("./lib/katex.js")
var inspect = require('util').inspect;
var latex = '\\sqrt { \\sin ( \\frac { \\Pi } { 2 } ) } = 1'
var tree = katex.__parse(latex)

function printTree(initialTree, printNode, getChildren) {
  function printBranch(tree, branch) {
    const isGraphHead = branch.length === 0;
    const children = getChildren(tree) || [];

    let branchHead = '';

    if (!isGraphHead) {
      branchHead = children && children.length !== 0 ? '┬ ' : '─ ';
    }

    const toPrint = printNode(tree, `${branch}${branchHead}`);

    if (typeof toPrint === 'string') {
      console.log(`${branch}${branchHead}${toPrint}`);
    }

    let baseBranch = branch;

    if (!isGraphHead) {
      const isChildOfLastBranch = branch.slice(-2) === '└─';
      baseBranch = branch.slice(0, -2) + (isChildOfLastBranch ? '  ' : '│ ');
    }

    const nextBranch = baseBranch + '├─';
    const lastBranch = baseBranch + '└─';

    children.forEach((child, index) => {
      printBranch(child, children.length - 1 === index ? lastBranch : nextBranch);
    });
  }

  printBranch(initialTree, '');
}

function getParseNode(tree) {
  if (!tree) return []
  if (tree.constructor.name == 'ParseNode') {
    return [tree]
  }
  if (typeof tree == 'object') {
    const res = []
    for (var i in tree) {
      res.push(...getParseNode(tree[i]))
    }
    return res
  }
  return []
}

//=============== 打印 ================
console.log(latex)
console.log("简单线条")
console.log(inspect(tree))

console.log("")
console.log("极致色彩")
printTree(
  {
    type: "root",
    value: tree,
    mode: ''
  },
  node => {
    if (typeof node.value == 'string') {
      return `${node.type} [${node.mode}, ${node.value}]`
    } else if (typeof node.value == 'object') {
      if (typeof node.value.body == 'string') {
        return `${node.type} [${node.mode}, ${node.value.body}]`
      }
      return `${node.type} [${node.mode}]`
    }
    return `${node.type}`
  },
  node => getParseNode(node.value)
);
//=========== OUTPUT ==================
/*
\sqrt { \sin ( \frac { \Pi } { 2 } ) } = 1
简单线条
[
  ParseNode {
    type: 'sqrt',
    value: { type: 'sqrt', body: [ParseNode], index: null },
    mode: 'math'
  },
  ParseNode { type: 'rel', value: '=', mode: 'math' },
  ParseNode { type: 'textord', value: '1', mode: 'math' }
]

极致色彩
root []
├─┬ sqrt [math]
│ └─┬ ordgroup [math]
│   ├── op [math, \sin]
│   ├── open [math, (]
│   ├─┬ genfrac [math]
│   │ ├─┬ ordgroup [math]
│   │ │ └── textord [math, \Pi]
│   │ └─┬ ordgroup [math]
│   │   └── textord [math, 2]
│   └── close [math, )]
├── rel [math, =]
└── textord [math, 1]
*/