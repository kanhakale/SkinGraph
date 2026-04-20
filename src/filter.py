class SemanticFilter:
    def __init__(self):
        self.toxin_keywords = [
            "pollutant", "emission", "smoke", "toxin", "toxicant", 
            "pesticide", "insecticide", "carcinogen", "effluent", 
            "waste", "exhaust", "particulate", "dioxin", "arsenic",
            "benzo(a)pyrene", "bisphenol", "phthalate"
        ]

    def is_safe(self, name):
        if not name or not isinstance(name, str):
            return False
        
        name_lower = name.lower()
        return not any(keyword in name_lower for keyword in self.toxin_keywords)

    def filter_hero_list(self, hero_rankings):
        return [item for item in hero_rankings if self.is_safe(item[0])]