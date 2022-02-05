importScripts("./pyodide/pyodide.js");

async function initialize() {
    self.pyodide = await loadPyodide({
        indexURL: './pyodide/'
    });
    self.pyodide.FS.writeFile('railroad.py', await (await fetch("railroad.py")).text());
    self.pyodide.FS.writeFile('plot.py', await (await fetch("plot.py")).text());
    self.plotPy = await pyodide.runPython(`
        import plot, sys

        def p(x):
            parts = []
            plot.plot_re(x).writeSvg(parts.append)
            return ''.join(parts)
        p
    `);

    postMessage({ 'type': 'initialized' });
}
let initialized = initialize();

onmessage = async function (e) {
    await initialized;
    try {
        self.postMessage({ 'type': 'plotted', 'diagram': self.plotPy(e.data) });
    } catch (e) {
        self.postMessage({
            'type': 'error',
            'msg': await this.self.pyodide.runPython('sys.last_value.args[0]')
        });
    }
};
