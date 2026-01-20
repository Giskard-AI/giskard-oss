# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import inspect

from datetime import datetime
import sys
from dataclasses import asdict

from sphinxawesome_theme import ThemeOptions
from sphinxawesome_theme.postprocess import Icons

html_permalinks_icon = Icons.permalinks_icon

project = "Giskard OSS"
copyright = f"{datetime.now().year}, Giskard"
author = "Giskard"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
    "sphinx.ext.githubpages",
    "sphinx_tabs.tabs",
    "sphinxext.opengraph",
    "notfound.extension",
]

templates_path = ["_templates"]
exclude_patterns = []

pygments_style = "lovelace"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinxawesome_theme"
html_static_path = ["_static"]
html_baseurl = "/"
source_suffix = [".rst", ".md"]

html_css_files = ["pygments-dark.css", "custom.css"]
html_js_files = ["custom.js"]
html_favicon = "_static/favicon.ico"

html_sidebars = {
    "**": [
        "sidebar_main_nav_links.html",
        "sidebars/sidebar_oss_checks.html",
    ],
}

theme_options = ThemeOptions(
    show_prev_next=True,
    show_scrolltop=True,
    awesome_external_links=True,
    logo_light="_static/logo_light.png",
    logo_dark="_static/logo_dark.png",
    main_nav_links={},
)
html_theme_options = asdict(theme_options)
# -- Open Graph configuration -------------------------------------------------
# https://sphinxext-opengraph.readthedocs.io/en/latest/

# Open Graph site name
ogp_site_name = "Giskard Documentation"
ogp_site_url = "https://docs.giskard.ai/"

# Open Graph image (logo for social sharing) - use relative path for local builds
ogp_image = "https://docs.giskard.ai/_static/open-graph-image.png"


# Add custom template function to render toctree from a specific document
def setup(app):
    def html_page_context(app, pagename, templatename, context, doctree):
        def toctree_from_doc(docname, **kwargs):
            """Render toctree starting from a specific document"""
            from sphinx.environment.adapters.toctree import TocTree
            from sphinx import addnodes

            source_doctree = app.env.get_doctree(docname)
            toctrees = list(source_doctree.findall(addnodes.toctree))

            if not toctrees:
                return ""

            toctree_adapter = TocTree(app.env)
            resolved = [
                toctree_adapter.resolve(
                    pagename,  # Use current page context, not the toctree source
                    app.builder,
                    toctree,
                    prune=False,
                    maxdepth=kwargs.get("maxdepth", -1),
                    titles_only=kwargs.get("titles_only", False),
                    collapse=kwargs.get("collapse", False),
                    includehidden=kwargs.get("includehidden", False),
                )
                for toctree in toctrees
            ]

            resolved = [r for r in resolved if r is not None]
            if not resolved:
                return ""

            result = resolved[0]
            for toctree in resolved[1:]:
                result.extend(toctree.children)

            return app.builder.render_partial(result)["fragment"]

        context["toctree_from_doc"] = toctree_from_doc

    app.connect("html-page-context", html_page_context)


# make github links resolve
def linkcode_resolve(domain, info):
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]
    print("##############")
    print(f"modname:{modname}")
    print(f"fullname:{fullname}")

    submod = sys.modules.get(modname)
    # print(submod)
    if submod is None:
        print("##############")

        return None
    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
            print(f"obj:{obj}")

            # print(obj)
        except:  # noqa: E722
            print("##############")
            return None

    try:
        fn = inspect.getsourcefile(
            obj.test_fn if hasattr(obj, "test_fn") else obj
        )  # TODO: generalise for other objects!
    # print(fn)
    except:  # noqa: E722
        fn = None
    if not fn:
        print("##############")

        return None
    print(f"fn:{fn}")

    try:
        source, lineno = inspect.getsourcelines(obj)
    except:  # noqa: E722
        lineno = None
        source = None

    if lineno and source:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""
    print(f"linespec:{linespec}")

    filename = fn.split("giskard_hub")[-1]
    print("##############")

    return f"https://github.com/Giskard-AI/giskard-hub/blob/main/src/giskard_hub{filename}{linespec}"


# Make 404 page work
notfound_urls_prefix = None
