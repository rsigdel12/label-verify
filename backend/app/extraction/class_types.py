import re
from enum import Enum

from rapidfuzz import fuzz


class AlcoholType(str, Enum):
    """Common TTB class/type families used for assisted matching."""

    VODKA = "Vodka"
    GIN = "Gin"
    RUM = "Rum"
    TEQUILA = "Tequila"
    WHISKEY = "Whiskey"
    BOURBON_WHISKEY = "Bourbon Whiskey"
    RYE_WHISKEY = "Rye Whiskey"
    SCOTCH_WHISKY = "Scotch Whisky"
    IRISH_WHISKEY = "Irish Whiskey"
    BRANDY = "Brandy"
    COGNAC = "Cognac"
    LIQUEUR = "Liqueur or Cordial"
    NEUTRAL_SPIRITS = "Neutral Spirits"
    RED_WINE = "Red Wine"
    WHITE_WINE = "White Wine"
    ROSE_WINE = "Rosé Wine"
    SPARKLING_WINE = "Sparkling Wine"
    TABLE_WINE = "Table Wine"
    DESSERT_WINE = "Dessert Wine"
    CIDER = "Cider"
    BEER = "Beer"
    LAGER = "Lager"
    ALE = "Ale"
    PORTER = "Porter"
    STOUT = "Stout"
    MALT_BEVERAGE = "Malt Beverage"
    SAKE = "Sake"


COMMON_ALCOHOL_TYPE_VALUES = tuple(item.value for item in AlcoholType)

_ALIASES: dict[AlcoholType, tuple[str, ...]] = {
    AlcoholType.VODKA: ("vodka", "flavored vodka"),
    AlcoholType.GIN: ("gin", "distilled gin", "london dry gin", "flavored gin"),
    AlcoholType.RUM: ("rum", "white rum", "dark rum", "spiced rum", "flavored rum"),
    AlcoholType.TEQUILA: ("tequila", "blanco tequila", "reposado tequila", "anejo tequila"),
    AlcoholType.BOURBON_WHISKEY: (
        "bourbon",
        "bourbon whiskey",
        "straight bourbon whiskey",
        "kentucky straight bourbon whiskey",
    ),
    AlcoholType.RYE_WHISKEY: ("rye whiskey", "straight rye whiskey"),
    AlcoholType.SCOTCH_WHISKY: ("scotch", "scotch whisky", "single malt scotch whisky"),
    AlcoholType.IRISH_WHISKEY: ("irish whiskey",),
    AlcoholType.WHISKEY: (
        "whiskey",
        "whisky",
        "corn whiskey",
        "malt whiskey",
        "blended whiskey",
    ),
    AlcoholType.COGNAC: ("cognac",),
    AlcoholType.BRANDY: ("brandy", "fruit brandy", "applejack"),
    AlcoholType.LIQUEUR: ("liqueur", "cordial", "creme liqueur"),
    AlcoholType.NEUTRAL_SPIRITS: ("neutral spirits", "grain spirits"),
    AlcoholType.RED_WINE: ("red wine", "cabernet sauvignon", "merlot", "pinot noir"),
    AlcoholType.WHITE_WINE: (
        "white wine",
        "chardonnay",
        "riesling",
        "sauvignon blanc",
        "pinot grigio",
    ),
    AlcoholType.ROSE_WINE: ("rose wine", "rosé wine"),
    AlcoholType.SPARKLING_WINE: ("sparkling wine", "champagne", "prosecco"),
    AlcoholType.TABLE_WINE: ("table wine", "light wine"),
    AlcoholType.DESSERT_WINE: ("dessert wine",),
    AlcoholType.CIDER: ("cider", "hard cider"),
    AlcoholType.LAGER: ("lager", "lager beer"),
    AlcoholType.ALE: ("ale", "pale ale", "india pale ale"),
    AlcoholType.PORTER: ("porter",),
    AlcoholType.STOUT: ("stout",),
    AlcoholType.MALT_BEVERAGE: ("malt beverage", "malt liquor", "near beer"),
    AlcoholType.BEER: ("beer",),
    AlcoholType.SAKE: ("sake",),
}

_FAMILIES = {
    AlcoholType.WHISKEY: "whiskey",
    AlcoholType.BOURBON_WHISKEY: "whiskey",
    AlcoholType.RYE_WHISKEY: "whiskey",
    AlcoholType.SCOTCH_WHISKY: "whiskey",
    AlcoholType.IRISH_WHISKEY: "whiskey",
    AlcoholType.BRANDY: "brandy",
    AlcoholType.COGNAC: "brandy",
    AlcoholType.BEER: "malt",
    AlcoholType.LAGER: "malt",
    AlcoholType.ALE: "malt",
    AlcoholType.PORTER: "malt",
    AlcoholType.STOUT: "malt",
    AlcoholType.MALT_BEVERAGE: "malt",
    AlcoholType.RED_WINE: "wine",
    AlcoholType.WHITE_WINE: "wine",
    AlcoholType.ROSE_WINE: "wine",
    AlcoholType.SPARKLING_WINE: "wine",
    AlcoholType.TABLE_WINE: "wine",
    AlcoholType.DESSERT_WINE: "wine",
}

_BROAD_TYPES = {
    AlcoholType.WHISKEY: {
        AlcoholType.BOURBON_WHISKEY,
        AlcoholType.RYE_WHISKEY,
        AlcoholType.SCOTCH_WHISKY,
        AlcoholType.IRISH_WHISKEY,
    },
    AlcoholType.BRANDY: {AlcoholType.COGNAC},
    AlcoholType.BEER: {
        AlcoholType.LAGER,
        AlcoholType.ALE,
        AlcoholType.PORTER,
        AlcoholType.STOUT,
    },
    AlcoholType.MALT_BEVERAGE: {
        AlcoholType.BEER,
        AlcoholType.LAGER,
        AlcoholType.ALE,
        AlcoholType.PORTER,
        AlcoholType.STOUT,
    },
}


def _normalize(value: str) -> str:
    normalized = value.casefold().replace("whisky", "whiskey").replace("é", "e")
    return " ".join(re.findall(r"[a-z0-9]+", normalized))


def classify_alcohol_type(value: str | None) -> tuple[AlcoholType | None, float]:
    """Map label text to a common type while tolerating small OCR errors."""
    if not value:
        return None, 0.0

    normalized = _normalize(value)
    aliases = sorted(
        (
            (_normalize(alias), alcohol_type)
            for alcohol_type, values in _ALIASES.items()
            for alias in values
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    broad_exact = None
    for alias, alcohol_type in aliases:
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            if len(alias.split()) > 1 or len(normalized.split()) == 1:
                return alcohol_type, 100.0
            broad_exact = alcohol_type
            break

    best_type = None
    best_score = 0.0
    for alias, alcohol_type in aliases:
        if len(alias) < 4:
            continue
        if broad_exact is not None and alcohol_type == broad_exact:
            continue
        score = float(fuzz.partial_ratio(alias, normalized))
        if score > best_score:
            best_type, best_score = alcohol_type, score
    if broad_exact is not None and best_score >= 92:
        return best_type, best_score
    if broad_exact is not None:
        return broad_exact, 100.0
    return (best_type, best_score) if best_score >= 82 else (None, best_score)


def same_type_family(first: AlcoholType | None, second: AlcoholType | None) -> bool:
    if first is None or second is None:
        return False
    return first == second or _FAMILIES.get(first, first.value) == _FAMILIES.get(
        second, second.value
    )


def expected_type_accepts(expected: AlcoholType | None, actual: AlcoholType | None) -> bool:
    """Return whether a broad selected enum type contains the detected subtype."""
    if expected is None or actual is None:
        return False
    return expected == actual or actual in _BROAD_TYPES.get(expected, set())
