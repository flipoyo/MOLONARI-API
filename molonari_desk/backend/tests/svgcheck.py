"""Contrôles partagés des SVG : textes d'une figure et estimation de leur étendue (rapport public de F1, figures de l'admin de F2)."""

from xml.etree import ElementTree


def svg_texts(report):
    for name, content in report.files.items():
        if name.endswith(".svg"):
            root = ElementTree.fromstring(content)
            yield name, root, ["".join(t.itertext()) for t in root.iter("{http://www.w3.org/2000/svg}text")]


def text_extents(root):
    """(texte, gauche, droite) estimés de chaque `<text>` d'un SVG, dans le repère de la racine : largeur = nombre de
    caractères × CHAR_WIDTH (×13/12 pour un titre), ancrage start / middle / end, `translate` des panneaux cumulés."""
    from molonari_twin.report.svgplot import CHAR_WIDTH

    svg = "{http://www.w3.org/2000/svg}"

    def walk(node, dx):
        if node.tag == f"{svg}g" and node.get("transform", "").startswith("translate("):
            dx += float(node.get("transform")[len("translate("):-1].split()[0])
        if node.tag == f"{svg}text":
            text = "".join(node.itertext())
            small = sum(len(t.text or "") for t in node.iter(f"{svg}tspan"))  # exposants : 9 px au lieu de 12
            width = (len(text) - small * 0.25) * CHAR_WIDTH * (13 / 12 if node.get("font-weight") == "bold" else 1)
            x = float(node.get("x")) + dx
            anchor = node.get("text-anchor", "start")
            left = {"start": x, "middle": x - width / 2, "end": x - width}[anchor]
            yield text, left, left + width
        for child in node:
            yield from walk(child, dx)

    return list(walk(root, 0.0))


def parse(content: bytes):
    """(racine, textes) d'un SVG donné en octets : l'adaptateur de `svg_texts` pour une figure hors rapport."""
    root = ElementTree.fromstring(content)
    return root, ["".join(t.itertext()) for t in root.iter("{http://www.w3.org/2000/svg}text")]
