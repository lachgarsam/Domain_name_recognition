NUM_SAMPLES = 500
DOMAINS_PER_SAMPLE = 5
OUTPUT_FILE = "synthetic_domain_dataset.json"

INDUSTRIES = {
    "Tech Startup": ["AI", "cloud", "data", "cyber", "code"],
    "E-commerce": ["shop", "cart", "deal", "market", "store"],
    "Food & Beverage": ["brew", "bean", "bite", "taste", "sip"],
    "Health & Wellness": ["fit", "zen", "vital", "heal", "yoga"],
    "Creative & Media": ["studio", "art", "design", "pix", "create"],
    "Local Business": ["auto", "fix", "wash", "clean", "home"],
    "Non-Profit & Education": ["learn", "help", "edu", "cause", "hope"],
}

PREFIXES = ["get", "try", "my", "go", "the", "smart", "easy", "eco"]
SUFFIXES = ["ly", "ify", "hub", "labs", "verse", "spot", "zone", "world"]
TLDS = [".com", ".co", ".io", ".ai", ".net"]
BLACKLIST = {"sex", "kill", "drugs", "hate", "nazi", "terror", "porn"}

# Foreign words and number tokens
FOREIGN_WORDS = ["vita", "mundo", "zen", "bello", "natura", "lumo", "kumo"]
NUM_TOKENS = ["24", "360", "101", "4U", "2Go", "365"]


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_NEW_TOKENS = 512
BATCH_SIZE = 4


# Prompt template
SYSTEM_PROMPT = """
You are a data-generation assistant. Your job is to invent realistic business profiles
and for each one suggest up to 3 original domain names, each with a confidence score between 0.00 and 1.00
indicating how good/attractive and memorable the domain is.
Domains should be plausible, available-looking, and aligned with the business’s branding, audience, and industry.
Higher confidence means a stronger, more compelling name.

Output each example as a single JSON object (JSON Lines format), with keys:

- "input": a one-sentence description of the business.
- "output": a list of domain suggestion objects, each with:
  - "domain": the domain string of format domain_name.tld
  - "confidence": a float from 0.00–1.00.

Do not output any extra text or commentary—only the JSON objects.

### Examples

{"input":"A vegan bakery specializing in artisanal sourdough loaves and organic pastries.","output":[{"domain":"GreenLoafBakery.com","confidence":0.92},{"domain":"VeganCrust.io","confidence":0.85},{"domain":"ArtisanVeganTreats.net","confidence":0.78}]}

{"input":"An on-demand mobile car detailing service that comes to your home or office within an hour.","output":[{"domain":"DetailDash.com","confidence":0.88},{"domain":"MobileShine.co","confidence":0.82},{"domain":"SparkleOnTheGo.com","confidence":0.75}]}

{"input":"A subscription box delivering rare and exotic houseplants directly to urban apartments.","output":[{"domain":"UrbanJungleBox.com","confidence":0.90},{"domain":"ExoticLeafClub.io","confidence":0.83},{"domain":"PlantParcel.net","confidence":0.77}]}

### Generate new examples below
"""
