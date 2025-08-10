import random

from config import INDUSTRIES, PREFIXES, SUFFIXES, TLDS, FOREIGN_WORDS, NUM_TOKENS
from utils import sanitize


def generate_prefix_suffix(keyword):
    """
    Generate a synthetic domain name by combining a given keyword
    with optional prefixes, suffixes, and a top-level domain (TLD).

    Parameters:
        keyword (str): A keyword related to the business or industry.

    Returns:
        str: A plausible domain name candidate.
    """
    prefix = random.choice(PREFIXES + [""])
    suffix = random.choice(SUFFIXES + [""])
    tld = random.choice(TLDS)
    core = keyword.capitalize()
    name = random.choice(
        [f"{prefix}{core}{suffix}", f"{core}{suffix}", f"{prefix}{core}"]
    )
    return sanitize(name) + tld


def generate_compound_word(keyword1, keyword2=None):
    """
    Generate a domain name by combining two keywords into a compound word,
    followed by a random TLD.

    Parameters:
        keyword1 (str): The first keyword (required).
        keyword2 (str, optional): The second keyword. If not provided,
                                  one is randomly selected from the INDUSTRIES vocabulary.

    Returns:
        str: A compound-style domain name (e.g., "DataBrew.io").
    """
    keyword2 = keyword2 or random.choice(sum(INDUSTRIES.values(), []))
    combined = keyword1.capitalize() + keyword2.capitalize()
    tld = random.choice(TLDS)
    return sanitize(combined) + tld


def generate_exotic_word(keyword):
    """
    Generate a domain name by combining the given keyword with a randomly selected
    foreign-sounding word to simulate an exotic brand name.

    Parameters:
        keyword (str): A business-related keyword.

    Returns:
        str: A domain name using an exotic naming pattern (e.g., "NekoZen.com").
    """
    foreign = random.choice(FOREIGN_WORDS)
    tld = random.choice(TLDS)
    name = random.choice(
        [
            foreign.capitalize() + keyword.capitalize(),
            keyword.capitalize() + foreign.capitalize(),
        ]
    )
    return sanitize(name) + tld


def generate_number_based(keyword):
    """
    Generate a domain name by combining the keyword with a number or numeric token.

    Parameters:
        keyword (str): A base keyword related to the business.

    Returns:
        str: A domain name with numeric branding (e.g., "Fit360.com").
    """
    token = random.choice(NUM_TOKENS)
    tld = random.choice(TLDS)
    name = random.choice(
        [f"{keyword.capitalize()}{token}", f"{token}{keyword.capitalize()}"]
    )
    return sanitize(name) + tld


def generate_misspelled(keyword):
    """
    Generate a domain name by applying simple character substitutions
    to simulate a deliberate brand-style misspelling.

    Parameters:
        keyword (str): The original keyword to be distorted.

    Returns:
        str: A misspelled version of the keyword with a TLD (e.g., "Kutz.io").
    """
    subs = {"c": "k", "i": "y", "s": "z", "o": "u"}
    result = "".join(subs.get(c, c) for c in keyword)
    result = result.capitalize()
    tld = random.choice(TLDS)
    return sanitize(result) + tld


def generate_acronym(industry, keyword):
    """
    Generate a domain name using the acronym of the industry and keyword,
    optionally adding a suffix and TLD.

    Parameters:
        industry (str): The name of the industry (e.g., "Tech Startup").
        keyword (str): A keyword to include in the acronym.

    Returns:
        str: An acronym-style domain name (e.g., "TSBrewHub.io").
    """
    words = industry.split() + [keyword]
    acronym = "".join(w[0].upper() for w in words if w)
    suffix = random.choice(SUFFIXES + [""])
    tld = random.choice(TLDS)
    return sanitize(acronym + suffix) + tld


def generate_brand_style():
    """
    Generate a brand-style domain name by combining random syllables to create
    a unique, catchy, and non-semantic name.

    Returns:
        str: A brand-like invented domain (e.g., "Zomora.io").
    """
    syllables = ["zo", "ka", "lu", "mo", "vi", "ra", "ne", "fy", "xo", "qi"]
    name = "".join(random.choices(syllables, k=random.randint(2, 3)))
    tld = random.choice(TLDS)
    return sanitize(name.capitalize()) + tld


# Strategy registry
STRATEGIES = [
    generate_prefix_suffix,
    generate_compound_word,
    generate_exotic_word,
    generate_number_based,
    generate_misspelled,
    generate_acronym,
    generate_brand_style,
]
