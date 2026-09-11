/* Turn the embedded marimo notebooks dark with the docs theme switch.
 *
 * The notebooks are exported with `display.theme = "system"` so they follow
 * `prefers-color-scheme`, but the theme switch only sets `data-theme` on this
 * document: browsers do not forward it into an iframe as a preferred colour
 * scheme. Marimo watches `data-vscode-theme-kind` on its body — the bridge
 * for hosts that embed it, taking precedence over its configured theme — so
 * this relays the resolved docs theme there.
 */
(function () {
    "use strict";
    function sync() {
        const kind = document.documentElement.dataset.theme === "dark"
            ? "vscode-dark" : "vscode-light";
        for (const frame of document.querySelectorAll(".marimo-notebook")) {
            const body = frame.contentDocument && frame.contentDocument.body;
            if (body) { body.dataset.vscodeThemeKind = kind; }
        }
    }
    new MutationObserver(sync).observe(
        document.documentElement,
        { attributes: true, attributeFilter: ["data-theme"] });
    document.addEventListener("DOMContentLoaded", function () {
        for (const frame of document.querySelectorAll(".marimo-notebook")) {
            frame.addEventListener("load", sync);
        }
        sync();
    });
})();
