# Abstract Syntax Tree for LaTeX

Light-weight AST forked from katex, modified.

## Usage

### Build Abstract Syntax Tree for LaTeX

```js
var katex = require("./lib/katex.js")
var latex = '\\sqrt { \\sin ( \\frac { \\Pi } { 2 } ) } = 1'
var tree = katex.__parse(latex)
```

## Other Usage

### In-browser rendering

Call `katex.render` with a TeX expression and a DOM element to render into:

```js
katex.render("c = \\pm\\sqrt{a^2 + b^2}", element);
```

If KaTeX can't parse the expression, it throws a `katex.ParseError` error.

### Server side rendering or rendering to a string

To generate HTML on the server or to generate an HTML string of the rendered math, you can use `katex.renderToString`:

```js
var html = katex.renderToString("c = \\pm\\sqrt{a^2 + b^2}");
// '<span class="katex">...</span>'
```

Make sure to include the CSS and font files, but there is no need to include the JavaScript. Like `render`, `renderToString` throws if it can't parse the expression.

### Rendering options

You can provide an object of options as the last argument to `katex.render` and `katex.renderToString`. Available options are:

- `displayMode`: `boolean`. If `true` the math will be rendered in display mode, which will put the math in display style (so `\int` and `\sum` are large, for example), and will center the math on the page on its own line. If `false` the math will be rendered in inline mode. (default: `false`)
- `throwOnError`: `boolean`. If `true`, KaTeX will throw a `ParseError` when it encounters an unsupported command. If `false`, KaTeX will render the unsupported command as text in the color given by `errorColor`. (default: `true`)
- `errorColor`: `string`. A color string given in the format `"#XXX"` or `"#XXXXXX"`. This option determines the color which unsupported commands are rendered in. (default: `#cc0000`)

For example:

```js
katex.render("c = \\pm\\sqrt{a^2 + b^2}", element, { displayMode: true });
```
