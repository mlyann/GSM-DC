import random

# No dot (".") is allowed in the parameter names, since it will be used in the parser.

class Data(object):
    def __init__(self) -> None:
        self.unique = []
        self.categ_list = [
            ["District", "Supermarket", "Product", "Ingredient"],
            ["Zoo", "Enclosure", "Animal", "Bone"],
            ["School", "Classroom", "Backpack", "Stationery"],
            ["Ecosystems", "Creatures", "Organs", "Cells"],
        ]

        self.categ_dict = {
            "District": {
                "Residential Districts": [
                    "Suburb", "Gated Community", "Apartment Complex", "Townhouse Development",
                    "Retirement Community", "Mobile Home Park", "Public Housing", "Condominium",
                    "Row Houses", "Single-Family Homes", "Duplexes", "Rural Homes",
                    "Bungalow Colony", "Cottage Community", "Golf Course Community", "Luxury Homes",
                    "Mountain Homes", "Lake Homes", "Beach Homes", "Farmhouses"
                ],
                "Commercial Districts": [
                    "Shopping District", "Business District", "Financial District", "Industrial District",
                    "Warehouse District", "Market District", "Restaurant District", "Entertainment District",
                    "Arts District", "Fashion District", "Silicon Valley", "Wall Street",
                    "Tech Park", "Automotive District", "Jewelry District", "Medical District",
                    "Legal District", "Media District", "Research Park", "Manufacturing District"
                ],
                "Historical Districts": [
                    "Old Town", "Historic Downtown", "Cultural Heritage District", "Archaeological District",
                    "Colonial District", "Victorian District", "Ancient City", "Preservation District",
                    "National Historic Landmark District", "Renaissance District", "Art Deco District", "Castle District",
                    "Monument Zone", "Cultural Landscape", "Quarters", "Old Harbor",
                    "Historic Industrial District", "Old Fortress", "Medieval Town", "Historic Plantation"
                ],
                "Educational Districts": [
                    "University District", "College Town", "School District", "Research District",
                    "Science Park", "Medical School Campus", "Engineering Campus", "Business School Campus",
                    "Arts Campus", "Vocational School District", "Preparatory School District", "Language School District",
                    "International School District", "Primary School District", "High School District", "Special Education District",
                    "Liberal Arts College District", "Community College District", "Technical College District", "Residential College District"
                ],
                "Government Districts": [
                    "Capitol District", "Embassy District", "Court District", "Administrative District",
                    "Municipal District", "Civic Center", "Federal District", "Parliamentary District",
                    "Presidential District", "Legislative District", "Judicial District", "Bureaucratic District",
                    "Consulate District", "Public Service District", "City Hall District", "Town Hall District",
                    "Diplomatic District", "Governor's District", "Military District", "Police District"
                ]
            },
            "Supermarket": { # Can online supermarket be included here?
                "General Supermarkets": [
                    "Walmart", "Tesco", "Carrefour", "Aldi",
                    "Lidl", "Kroger", "Safeway", "Asda",
                    "Morrisons", "Publix", "Waitrose", "Sainsbury's",
                    "Costco", "Sam's Club", "BJ's Wholesale Club", "WinCo Foods",
                    "Giant Eagle", "Meijer", "Wegmans", "Albertsons"
                ],
                "Health Food Supermarkets": [
                    "Whole Foods Market", "Trader Joe's", "Sprouts Farmers Market", "The Fresh Market",
                    "Erewhon Market", "Natural Grocers", "Lassens Natural Foods", "Earth Fare",
                    "Fresh Thyme Market", "New Seasons Market", "MOM's Organic Market", "Jungle Jim's International Market",
                    "PCC Community Markets", "Lazy Acres Natural Market", "Nature's Food Patch", "Harvest Market",
                    "Dean's Natural Food Market", "Jimbo's Naturally", "Lakewinds Food Co-op", "BriarPatch Food Co-op"
                ],
                "International Supermarkets": [
                    "H Mart", "99 Ranch Market", "Zion Market", "Seafood City Supermarket",
                    "Uwajimaya", "Mitsuwa Marketplace", "Marukai Corporation", "Asia Supermarket",
                    "Fiesta Mart", "Superior Grocers", "Valli Produce", "Patel Brothers",
                    "T&T Supermarket", "Galleria Supermarket", "Seabra Foods", "NetCost Market",
                    "Buford Highway Farmers Market", "New Grand Mart", "Sedano's", "La Michoacana Meat Market"
                ],
                "Discount Supermarkets": [
                    "Aldi", "Lidl", "Save-A-Lot", "Food 4 Less",
                    "WinCo Foods", "Grocery Outlet", "Dollar General", "Dollar Tree",
                    "Family Dollar", "Market Basket", "Price Rite", "Smart & Final",
                    "No Frills", "Price Chopper", "Marc's", "Shop 'n Save",
                    "Shoppers Value Foods", "Fiesta Foods", "Super Saver", "Save Mart"
                ],
                "Online Supermarkets": [
                    "Amazon Fresh", "Instacart", "FreshDirect", "Shipt",
                    "Peapod", "Google Express", "Boxed", "Jet",
                    "Ocado", "Postmates", "Uber Eats", "Deliveroo",
                    "HelloFresh", "Blue Apron", "Thrive Market", "Sun Basket",
                    "GrubMarket", "DoorDash", "Grofers", "Bigbasket"
                ]
            },
            "Product": {
                "Canned Foods": [
                    "Canned Beans", "Canned Vegetables", "Canned Fruits", "Canned Fish", 
                    "Canned Meats", "Canned Soups", "Canned Sauces", "Canned Pasta",
                    "Canned Milk", "Canned Broth", "Canned Tomatoes", "Canned Olives",
                    "Canned Peaches", "Canned Tuna", "Canned Chicken", "Canned Beef", 
                    "Canned Corn", "Canned Peas", "Canned Pineapple", "Canned Cherries"
                ],
                "Snack Foods": [
                    "Potato Chips", "Pretzels", "Popcorn", "Candy Bars", 
                    "Gummy Candy", "Cookies", "Crackers", "Granola Bars",
                    "Fruit Snacks", "Cheese Puffs", "Nuts", "Trail Mix",
                    "Beef Jerky", "Rice Cakes", "Yogurt Covered Raisins", "Chocolate Covered Pretzels", 
                    "Tortilla Chips", "Salsa", "Hummus", "Dried Fruit"
                ],
                "Beverages": [
                    "Bottled Water", "Juices", "Sodas", "Sports Drinks",
                    "Energy Drinks", "Tea", "Coffee", "Milk",
                    "Almond Milk", "Soy Milk", "Oat Milk", "Rice Milk",
                    "Hot Chocolate", "Wine", "Beer", "Liquor",
                    "Smoothies", "Protein Shakes", "Kombucha", "Sparkling Water"
                ],
                "Baked Goods": [
                    "Bread", "Bagels", "Cakes", "Pies",
                    "Donuts", "Pastries", "Cookies", "Rolls",
                    "Buns", "Croissants", "Muffins", "Pretzels",
                    "Tortillas", "Pita Bread", "Rye Bread", "Sourdough Bread", 
                    "Wheat Bread", "White Bread", "Cinnamon Rolls", "Danishes"
                ],
                "Dairy Products": [
                    "Milk", "Cheese", "Yogurt", "Butter", 
                    "Cream", "Ice Cream", "Sour Cream", "Cottage Cheese",
                    "Whipped Cream", "Cream Cheese", "Cheddar Cheese", "Mozzarella Cheese",
                    "Brie Cheese", "Blue Cheese", "Swiss Cheese", "Parmesan Cheese", 
                    "Goat Cheese", "Ricotta Cheese", "Provolone Cheese", "Monterey Jack Cheese"
                ]
            },
            "Ingredient": {
                "Spices and Herbs": [
                    "Basil", "Oregano", "Cumin", "Paprika",
                    "Thyme", "Garlic Powder", "Onion Powder", "Rosemary",
                    "Dill", "Coriander", "Turmeric", "Chili Powder",
                    "Saffron", "Parsley", "Cayenne Pepper", "Cinnamon",
                    "Nutmeg", "Cloves", "Cardamom", "Mint"
                ],
                "Vegetables": [
                    "Tomato", "Cucumber", "Carrot", "Onion", 
                    "Bell Pepper", "Garlic", "Potato", "Eggplant",
                    "Lettuce", "Spinach", "Broccoli", "Mushrooms",
                    "Zucchini", "Cabbage", "Celery", "Peas",
                    "Sweet Corn", "Beetroot", "Asparagus", "Kale"
                ],
                "Fruits": [
                    "Apple", "Banana", "Orange", "Lemon",
                    "Strawberry", "Pineapple", "Mango", "Grape",
                    "Cherry", "Peach", "Pear", "Watermelon",
                    "Kiwi", "Pomegranate", "Blueberry", "Raspberry",
                    "Coconut", "Lime", "Grapefruit", "Avocado"
                ],
                "Grains and Cereals": [
                    "Rice", "Wheat", "Corn", "Barley", 
                    "Oats", "Rye", "Millet", "Sorghum",
                    "Quinoa", "Buckwheat", "Amaranth", "Teff",
                    "Spelt", "Brown Rice", "Whole Wheat", "White Rice",
                    "Wild Rice", "Rolled Oats", "Steel-Cut Oats", "Popcorn"
                ],
                "Meats and Poultry": [
                    "Chicken", "Beef", "Pork", "Lamb", 
                    "Turkey", "Duck", "Bacon", "Sausage",
                    "Ham", "Venison", "Salami", "Pepperoni",
                    "Chicken Breast", "Ground Beef", "Pork Chop", "Lamb Chop",
                    "Beef Steak", "Chicken Wing", "Turkey Breast", "Duck Confit"
                ]
            },
            "Zoo": {
                "Traditional Zoos": [
                    "San Diego Zoo", "Bronx Zoo", "London Zoo", "Berlin Zoological Garden", 
                    "Beijing Zoo", "Toronto Zoo", "Melbourne Zoo", "Edinburgh Zoo",
                    "Sydney Zoo", "Moscow Zoo", "Singapore Zoo", "Zoo de Madrid",
                    "Houston Zoo", "Taronga Zoo", "Philadelphia Zoo", "Zoo Miami",
                    "Cincinnati Zoo", "Detroit Zoo", "Calgary Zoo", "Zoo Zürich"
                ],
                "Aquariums": [
                    "Georgia Aquarium", "Monterey Bay Aquarium", "Oceanografic", "Okinawa Churaumi Aquarium", 
                    "The Florida Aquarium", "National Aquarium", "Seattle Aquarium", "The Deep",
                    "Shedd Aquarium", "Sydney Aquarium", "Vancouver Aquarium", "Two Oceans Aquarium",
                    "Tennessee Aquarium", "Ripley's Aquarium of Canada", "Aquarium of the Pacific", "Dallas World Aquarium",
                    "Baltimore Aquarium", "Adventure Aquarium", "SeaWorld Orlando", "The Living Seas"
                ],
                "Safari Parks": [
                    "San Diego Zoo Safari Park", "Kruger National Park", "Serengeti National Park", "Yala National Park", 
                    "Hluhluwe-Imfolozi Park", "Masai Mara", "Amboseli National Park", "Tarangire National Park",
                    "Lake Nakuru National Park", "Etosha National Park", "Pilanesberg National Park", "Tsavo East National Park",
                    "Selous Game Reserve", "Ngorongoro Conservation Area", "Moremi Game Reserve", "Mikumi National Park",
                    "Kgalagadi Transfrontier Park", "Chobe National Park", "Addo Elephant National Park", "Ruaha National Park"
                ],
                "Aviaries": [
                    "Jurong Bird Park", "Kuala Lumpur Bird Park", "Monte Casino Bird Park", "Birds of Eden", 
                    "Oklahoma City Zoo and Botanical Garden", "Living Desert Zoo and Gardens", "Tracy Aviary", "Niagara Falls Aviary",
                    "National Aviary", "Avery Island", "Lotherton Hall Bird Garden", "Flamingo Gardens",
                    "World of Birds", "Birdland Park and Gardens", "Birdworld", "Sylvan Heights Bird Park",
                    "Bird Paradise Wildlife Park", "Bird Kingdom", "San Antonio Zoo", "Pittsburgh National Aviary"
                ],
                "Insectariums": [
                    "Montreal Insectarium", "Insectarium de Montréal", "Audubon Insectarium", "Insectarium of Victoria", 
                    "Cleveland Insectarium", "Insectarium of Philadelphia", "Detroit Insectarium", "Insectarium of Chicago",
                    "Houston Insectarium", "Los Angeles Insectarium", "San Francisco Insectarium", "Insectarium of Boston",
                    "San Diego Insectarium", "Insectarium of Denver", "Phoenix Insectarium", "Insectarium of Seattle",
                    "San Jose Insectarium", "Insectarium of Dallas", "Austin Insectarium", "Insectarium of Washington DC"
                ]
            },
            "Enclosure": {
                "Mammal Enclosures": [
                    "Lion Exhibit", "Elephant Habitat", "Giraffe Enclosure", "Primate House",
                    "Bear Den", "Tiger Territory", "Kangaroo Walkabout", "Cheetah Run",
                    "Wolf Woods", "Panda Exhibit", "Hippopotamus Tank", "Rhinoceros Pen",
                    "Bat Cave", "Small Mammal House", "Otter Pool", "Meerkat Manor",
                    "Camel Yard", "Zebra Paddock", "Lemur Island", "Anteater Area"
                ],
                "Bird Enclosures": [
                    "Aviary", "Penguin Exhibit", "Parrot Paradise", "Eagle Eyrie",
                    "Flamingo Lagoon", "Owl Forest", "Peacock Garden", "Falcon Aviary",
                    "Toucan Terrace", "Pheasant Pen", "Duck Pond", "Pelican Pier",
                    "Crane Courtyard", "Hawk Haven", "Pigeon Loft", "Vulture Valley",
                    "Swan Lake", "Macaw Island", "Condor Cliff", "Kiwi House"
                ],
                "Reptile and Amphibian Enclosures": [
                    "Reptile House", "Crocodile Swamp", "Turtle Pond", "Snake Den",
                    "Lizard Lounge", "Frog Bog", "Alligator Alley", "Iguana Habitat",
                    "Salamander Stream", "Gecko Greenhouse", "Tortoise Garden", "Chameleon Cage",
                    "Toad Abode", "Komodo Dragon Den", "Python Palace", "Newt Nook",
                    "Rattlesnake Ravine", "Monitor Lizard Habitat", "Caiman Creek", "Anaconda Exhibit"
                ],
                "Aquatic Enclosures": [
                    "Aquarium", "Shark Tank", "Coral Reef Exhibit", "Jellyfish Gallery",
                    "Seahorse Sanctuary", "Ray Pool", "Octopus Den", "Freshwater Fish Exhibit",
                    "Saltwater Fish Exhibit", "Seal Pool", "Dolphin Lagoon", "Kelp Forest Tank",
                    "Rockpool Exhibit", "Tropical Fish Tanks", "Sealion Show Pool", "Penguin Beach",
                    "Manatee Mangrove", "Crustacean Cove", "Seadragon Deep", "Piranha Pond"
                ],
                "Insect and Invertebrate Enclosures": [
                    "Insect House", "Butterfly Garden", "Spider Web Exhibit", "Ant Farm",
                    "Beetle Bungalow", "Worm World", "Mantis Habitat", "Honeybee Hive",
                    "Scorpion Den", "Tarantula Terrarium", "Caterpillar Nursery", "Cricket Chorus",
                    "Millipede Meadow", "Ladybug Loft", "Dragonfly Delta", "Stick Insect Forest",
                    "Snail Shellter", "Cockroach Corner", "Leafcutter Ant Exhibit", "Moth Metropolis"
                ]
            },
            "Animal": {
                "Mammals": [
                    "Elephant", "Tiger", "Bear", "Lion", 
                    "Kangaroo", "Giraffe", "Zebra", "Monkey",
                    "Koala", "Panda", "Hippopotamus", "Rhinoceros",
                    "Leopard", "Cheetah", "Wolf", "Otter",
                    "Deer", "Camel", "Raccoon", "Squirrel"
                ],
                "Birds": [
                    "Eagle", "Parrot", "Swan", "Peacock", 
                    "Pigeon", "Sparrow", "Penguin", "Ostrich",
                    "Falcon", "Hawk", "Hummingbird", "Flamingo",
                    "Crow", "Owl", "Canary", "Finch",
                    "Woodpecker", "Albatross", "Seagull", "Dove"
                ],
                "Reptiles": [
                    "Snake", "Crocodile", "Turtle", "Lizard", 
                    "Alligator", "Iguana", "Chameleon", "Gecko",
                    "Komodo Dragon", "Rattlesnake", "Tortoise", "Anaconda",
                    "Gila Monster", "Monitor Lizard", "Boa Constrictor", "Python",
                    "Skink", "Anole", "Frilled Lizard", "Bearded Dragon"
                ],
                "Amphibians": [
                    "Frog", "Toad", "Salamander", "Newt", 
                    "Caecilian", "Tree Frog", "Bullfrog", "Poison Dart Frog",
                    "Pacman Frog", "Fire Salamander", "Axolotl", "Red Eyed Tree Frog",
                    "Mudskipper", "Hellbender", "Giant Salamander", "Glass Frog",
                    "Siren", "Cave Salamander", "Spotted Salamander", "Tiger Salamander"
                ],
                "Fish": [
                    "Goldfish", "Shark", "Tuna", "Salmon", 
                    "Trout", "Bass", "Swordfish", "Mackerel",
                    "Carp", "Pike", "Cod", "Herring",
                    "Anchovy", "Eel", "Flounder", "Haddock",
                    "Halibut", "Mahi Mahi", "Marlin", "Sardine"
                ]
            },
            "Bone": {
                "Bones in a Bird's Wing": [
                    "Humerus", "Radius", "Ulna", "Carpometacarpus",
                    "Alula", "Primary Feathers", "Secondary Feathers", "Tertials",
                    "Scapula", "Coracoid", "Furcula", "Proximal Phalanges",
                    "Middle Phalanges", "Distal Phalanges", "Metacarpals"
                ],
                "Bones in a Cat's Paw": [
                    "Radial Carpal", "Intermediate Carpal", "Ulnar Carpal", "Accessory Carpal",
                    "Metacarpal I", "Metacarpal II", "Metacarpal III", "Metacarpal IV", 
                    "Metacarpal V", "Proximal Phalanx", "Middle Phalanx", "Distal Phalanx"
                ],
                "Bones in a Dog's Tail": [
                    "Caudal Vertebrae 1", "Caudal Vertebrae 2", "Caudal Vertebrae 3",
                    "Caudal Vertebrae 4", "Caudal Vertebrae 5", "Caudal Vertebrae 6",
                    "Caudal Vertebrae 7", "Caudal Vertebrae 8", "Caudal Vertebrae 9",
                    "Caudal Vertebrae 10", "Caudal Vertebrae 11", "Caudal Vertebrae 12"
                ],
                "Bones in a Horse's Leg": [
                    "Scapula", "Humerus", "Radius", "Ulna",
                    "Carpal Bones", "Metacarpal Bones", "Proximal Phalanges",
                    "Middle Phalanges", "Distal Phalanges", "Sesamoid Bones",
                    "Cannon Bone", "Pastern Bone", "Coffin Bone", "Fetlock"
                ],
                "Bones in a Fish's Fin": [
                    "Pectoral Fin Rays", "Pelvic Fin Rays", "Dorsal Fin Rays",
                    "Caudal Fin Rays", "Anal Fin Rays", "Fin Spines", "Proximal Radials",
                    "Distal Radials", "Basal Plate", "Epicentrals", "Hypurals",
                    "Prezygapophyses", "Postzygapophyses", "Neural Spines", "Haemal Spines"
                ]
            },
            "School": {
                "Elementary Schools": [
                    "Maple Grove Elementary", "Sunset Primary", "Pine Ridge Elementary", "River Valley Elementary", "Hilltop Elementary", 
                    "Meadowbrook Primary", "Cedarwood Elementary", "North Star Elementary", "Brookside Elementary", "Green Field Elementary", 
                    "Sunnydale Primary", "Mountain Peak Elementary", "Lakeview Elementary", "Springfield Primary", "Woodland Elementary", 
                    "Pleasant Hill Primary", "Westridge Elementary", "Elmwood Elementary", "Forest Glen Primary", "Brightside Elementary"
                ],
                "High Schools": [
                    "Lincoln High", "Riverview High", "Crestwood Secondary", "Northridge High", "Eastside Secondary", 
                    "Westwood High", "Central High", "Southgate Secondary", "Pinnacle High", "Lakeshore High", 
                    "Hillcrest High", "Parkview Secondary", "Metropolitan High", "Valley View High", "Kingston Secondary", 
                    "Rosewood High", "Summit Secondary", "Greenwood High", "Liberty High", "Harborview High"
                ],
                "Colleges": [
                    "Pine Valley College", "Riverdale Community College", "East Bay College", "Westland College", "North Point College", 
                    "Sunrise College", "Lakefront Community College", "Golden Gate College", "Crescent College", "Metropolis Community College", 
                    "Harbor Point College", "Skyline College", "Meadowland College", "Midtown Community College", "Starlight College", 
                    "Riverside College", "Mountainview College", "Seaside Community College", "Woodgrove College", "Centennial College"
                ],
                "Universities": [
                    "University of Green Hills", "Riverstone University", "King's Crown University", "Pinnacle Peak University", "Lakeside University", 
                    "Metropolitan University", "Starlight University", "Crestwood University", "Sunnydale University", "Rosewood University", 
                    "Valley Forge University", "Eastwood University", "Westbrook University", "North Star University", "Seaview University", 
                    "Harbortown University", "Meadowland University", "Skyview University", "Forest Ridge University", "Summit University"
                ],
                "Online Schools": [
                    "EduWeb Academy", "NetLearn Institute", "DigitalEd School", "WebStudy University", "Online Academy", 
                    "CyberScholars", "E-Learn Portal", "GlobalNet University", "WebEd Institute", "Virtual Scholars Academy", 
                    "NetVersity", "WorldWide Web University", "EduLink Online", "StudyNet College", "WebWorld Institute", 
                    "CyberAcademy", "E-Study University", "WebMasters Academy", "NetEdge College", "Digital Scholars"
                ]
            },
            "Classroom": {
                "Science Laboratories": [
                    "Physics Lab", "Chemistry Lab", "Biology Lab", "Botany Lab", "Zoology Lab", 
                    "Astronomy Lab", "Microbiology Lab", "Genetics Lab", "Earth Science Lab", "Anatomy Lab",
                    "Ecology Lab", "Neuroscience Lab", "Molecular Biology Lab", "Geology Lab", "Organic Chemistry Lab", 
                    "Inorganic Chemistry Lab", "Physical Chemistry Lab", "Environmental Science Lab", "Computer Science Lab", "Forensic Lab"
                ],
                "Arts & Humanities Classrooms": [
                    "Visual Arts Studio", "Drama Studio", "Music Room", "Dance Studio", "Literature Classroom", 
                    "History Classroom", "Philosophy Classroom", "Language Classroom", "Anthropology Classroom", "Sociology Classroom",
                    "Graphic Design Studio", "Photography Studio", "Film Studio", "Ceramics Studio", "Drawing Studio",
                    "Theater Arts Classroom", "Geography Classroom", "Fashion Design Studio", "Digital Arts Studio", "Cultural Studies Classroom"
                ],
                "Math & Engineering Rooms": [
                    "Calculus Classroom", "Geometry Room", "Statistics Lab", "Engineering Workshop", "Robotics Lab", 
                    "Algebra Classroom", "Trigonometry Room", "Civil Engineering Lab", "Mechanical Engineering Lab", "Electrical Engineering Lab",
                    "Aerospace Engineering Lab", "Computer Engineering Lab", "Structural Engineering Lab", "Thermodynamics Lab", "Fluid Mechanics Lab",
                    "Materials Science Lab", "Control Systems Lab", "Differential Equations Classroom", "Linear Algebra Room", "Number Theory Room"
                ],
                "Social Sciences Rooms": [
                    "Psychology Lab", "Economics Classroom", "Political Science Room", "Public Administration Classroom", "Counseling Room", 
                    "International Relations Classroom", "Education Studies Room", "Behavioral Science Lab", "Criminology Classroom", "Sociology Lab",
                    "Social Work Classroom", "Geography Lab", "Anthropology Lab", "Archeology Lab", "Human Development Classroom",
                    "Communication Studies Room", "Journalism Lab", "Public Relations Studio", "Marketing Classroom", "Business Management Room"
                ],
                "Physical Education Areas": [
                    "Gymnasium", "Swimming Pool", "Martial Arts Dojo", "Dance Hall", "Yoga Studio", 
                    "Weight Training Room", "Aerobics Studio", "Basketball Court", "Volleyball Court", "Badminton Court",
                    "Tennis Court", "Table Tennis Room", "Rock Climbing Wall", "Archery Range", "Squash Court",
                    "Football Field", "Baseball Field", "Track and Field", "Golf Practice Area", "Hockey Rink"
                ]
            },
            "Backpack": {
                "Travel Backpacks": [
                    "Hiking Backpack", "Weekender Backpack", "Travel Daypack", "Backpacking Pack", "Carry-On Backpack",
                    "Trekking Backpack", "Ultralight Backpack", "Rucksack", "Mountaineering Backpack", "Duffle Backpack",
                    "Roll-Top Backpack", "Overnight Backpack", "Expedition Pack", "Convertible Backpack", "Multi-Day Pack",
                    "Wheeled Backpack", "Camping Backpack", "Top-Loading Backpack", "Compression Backpack", "Packable Travel Backpack"
                ],
                "School/College Backpacks": [
                    "Laptop Backpack", "Bookbag", "Rolling Backpack", "School Daypack", "Messenger Backpack",
                    "Canvas Backpack", "Printed Backpack", "Athletic Backpack", "Sling Backpack", "Scholar Backpack",
                    "Multi-Pocket Backpack", "Sturdy Backpack", "Waterproof School Backpack", "Anti-Theft Backpack", "Organizer Backpack",
                    "Two-Strap Backpack", "Casual Backpack", "Student Rucksack", "Fashion School Backpack", "Padded Backpack"
                ],
                "Sports and Fitness Backpacks": [
                    "Gym Backpack", "Skiing Backpack", "Running Backpack", "Biking Backpack", "Hydration Backpack",
                    "Yoga Mat Backpack", "Climbing Backpack", "Snowboard Backpack", "Surfing Backpack", "Golf Backpack",
                    "Swimming Backpack", "Training Daypack", "Sports Ball Backpack", "Shoe Compartment Backpack", "Fitness Backpack",
                    "Workout Backpack", "Racquet Backpack", "Ventilated Backpack", "Crossfit Backpack", "Sweat-Resistant Backpack"
                ],
                "Professional and Work Backpacks": [
                    "Laptop Professional Backpack", "Briefcase Backpack", "Camera Backpack", "Drone Backpack", "Tool Backpack",
                    "Office Backpack", "Designer Backpack", "Business Daypack", "Executive Backpack", "Tech Backpack",
                    "Presentation Backpack", "Organizer Work Backpack", "Slim Work Backpack", "Expandable Backpack", "Hardshell Backpack",
                    "Work Rucksack", "Modular Backpack", "Commuter Backpack", "Conference Backpack", "Daily Work Backpack"
                ],
                "Specialized and Miscellaneous Backpacks": [
                    "Diaper Backpack", "Tactical Backpack", "Solar Charging Backpack", "Hunting Backpack", "Fishing Backpack",
                    "First Aid Backpack", "Emergency Backpack", "Insulated Backpack", "Picnic Backpack", "Beach Backpack",
                    "Drawstring Backpack", "Lumbar Pack", "Fashion Backpack", "Clear Backpack", "Gaming Backpack",
                    "Concealed Carry Backpack", "Crafting Backpack", "Musical Instrument Backpack", "Astronomy Gear Backpack", "Toy Backpack"
                ]
            },
            "Stationery": {
                "Writing Instruments": [
                    "Ballpoint Pen", "Fountain Pen", "Gel Pen", "Rollerball Pen", "Highlighter",
                    "Permanent Marker", "Whiteboard Marker", "Pencil", "Mechanical Pencil", "Colored Pencil",
                    "Crayon", "Pastel", "Charcoal Stick", "Fineliner", "Brush Pen",
                    "Calligraphy Pen", "Erasable Pen", "Text Marker", "Oil Pastel", "Watercolor Pencil"
                ],
                "Paper Products": [
                    "Notebook", "Notepad", "Loose Leaf Paper", "Graph Paper", "Construction Paper",
                    "Sketchbook", "Diary", "Journal", "Post-it Notes", "Index Cards",
                    "Photocopy Paper", "Tracing Paper", "Carbon Paper", "Legal Pad", "Binder Paper",
                    "Flip Chart", "Craft Paper", "Colored Paper", "Scrapbooking Paper", "Wrapping Paper"
                ],
                "Office Supplies": [
                    "Stapler", "Paper Clip", "Rubber Band", "Binder Clip", "Hole Puncher",
                    "Sticky Tape", "Scissors", "Glue Stick", "Correction Fluid", "Correction Tape",
                    "Envelope", "Stamp Pad", "Ink Refill", "Laminating Sheets", "Calculator",
                    "Ruler", "Sticky Tabs", "Label Maker", "Push Pins", "Paperweight"
                ],
                "Organizational Tools": [
                    "Binder", "Folder", "Document Wallet", "File Organizer", "Ring Binder",
                    "Accordion File", "Card Holder", "Receipt Organizer", "Desk Organizer", "Magazine Holder",
                    "Clip Board", "Name Badge", "Business Card Holder", "Divider Tabs", "Calendar",
                    "Planner", "Desk Pad", "Box File", "Envelope Folder", "Document Box"
                ],
                "Art Supplies": [
                    "Acrylic Paint", "Watercolor Paint", "Paintbrush", "Palette", "Canvas Board",
                    "Easel", "Gouache", "Ink Bottle", "Spray Paint", "Stencil",
                    "Masking Tape", "Glitter", "Beads", "Clay", "Modelling Tools",
                    "Craft Knife", "Drawing Board", "Ribbon", "Stamp", "Embossing Powder"
                ]
            },
            "Ecosystems": {
                "Forest Ecosystems": [
                    "Tropical Rainforest", "Temperate Deciduous Forest", "Boreal Forest", 
                    "Mangrove Forest", "Subtropical Rainforest", "Temperate Evergreen Forest", 
                    "Cloud Forest", "Montane Forest", "Mixed Forest", "Coniferous Forest", 
                    "Taiga Forest", "Tropical Monsoon Forest", "Tropical Dry Forest", 
                    "Secondary Forest", "Old-growth Forest", "Riparian Forest", 
                    "Temperate Broadleaf Forest", "Pine Forest", "Spruce-Fir Forest", 
                    "Rainforest"]
                ,
                "Aquatic Ecosystems": [
                    "Coral Reef", "Open Ocean", "Estuary", "Freshwater Lake", 
                    "River Ecosystem", "Mangrove Ecosystem", "Kelp Forest", "Salt Marsh", 
                    "Brackish Water", "Seagrass Meadows", "Tidal Pool", "Deep Sea Ecosystem", 
                    "Freshwater Pond", "Stream Ecosystem", "Wetland", "Hydrothermal Vent", 
                    "Pelagic Zone", "Benthic Zone", "Coastal Ecosystem", "Arctic Ocean"]
                ,
                "Grassland Ecosystems": [
                    "Savanna", "Prairie", "Steppe", "Pampas", "Veldt", 
                    "Temperate Grassland", "Tropical Grassland", "Highland Grassland", "Flooded Grassland", 
                    "Montane Grassland", "Lowland Grassland", "Desert Fringe Grassland", 
                    "Meadow", "Alpine Grassland", "Tallgrass Prairie", "Shortgrass Prairie", 
                    "Mixed Grass Prairie", "Coastal Grassland", "Subtropical Grassland", "Northern Grassland"]
                ,
                "Desert Ecosystems": [
                    "Sahara Desert", "Arabian Desert", "Gobi Desert", "Kalahari Desert", 
                    "Great Victoria Desert", "Sonoran Desert", "Mojave Desert", "Chihuahuan Desert", 
                    "Thar Desert", "Karakum Desert", "Dasht-e Kavir", "Atacama Desert", 
                    "Colorado Plateau Desert", "Great Basin Desert", "Namib Desert", 
                    "Patagonian Desert", "Syrian Desert", "Dasht-e Lut", "Negev Desert", 
                    "Turkestan Desert"]
                ,
                "Urban Ecosystems": [
                    "Central Park in New York City", "Hyde Park in London", "Ueno Park in Tokyo", 
                    "Golden Gate Park in San Francisco", "Griffith Park in Los Angeles", 
                    "Chapultepec Park in Mexico City", "Tiergarten in Berlin", "Vondelpark in Amsterdam", 
                    "Stanley Park in Vancouver", "Bois de Boulogne in Paris", "Gardens by the Bay in Singapore", 
                    "Lumphini Park in Bangkok", "Kings Park in Perth", "Yoyogi Park in Tokyo", 
                    "The High Line in New York City", "Millennium Park in Chicago", 
                    "Ibirapuera Park in São Paulo", "Bukhansan National Park in Seoul", 
                    "Hampstead Heath in London", "Lumpini Park in Bangkok"]
            },
            "Creatures": {
                "Terrestrial Animals": [
                    "African Elephant", "Bengal Tiger", "Grizzly Bear", "Kangaroo",
                    "Giraffe", "Panda", "Lion", "Wolf", "Cheetah", "Zebra",
                    "Hippopotamus", "Rhinoceros", "Gorilla", "Chimpanzee", "Moose",
                    "Jaguar", "Leopard", "Sloth", "Anteater", "Armadillo"],
                "Aquatic Animals": [
                    "Blue Whale", "Great White Shark", "Dolphin", "Octopus",
                    "Clownfish", "Seahorse", "Stingray", "Jellyfish", "Sea Turtle",
                    "Walrus", "Starfish", "Sea Urchin", "Manatee", "Lobster", "Crab",
                    "Coral", "Moray Eel", "Manta Ray", "Puffer Fish", "Sea Anemone"],
                "Aerial Animals": [
                    "Bald Eagle", "Peregrine Falcon", "Hummingbird", "Albatross",
                    "Kingfisher", "Owl", "Parrot", "Flamingo", "Swan", "Vulture",
                    "Woodpecker", "Peacock", "Stork", "Pigeon", "Bat",
                    "Butterfly", "Dragonfly", "Hawk", "Condor", "Crow"],
                "Microscopic Organisms": [
                    "Amoeba", "Paramecium", "Euglena", "Volvox", "Stentor",
                    "Diatom", "Rotifer", "Tardigrade", "Plankton", "Yeast",
                    "E coli", "Staphylococcus", "Chlamydomonas", "Spirogyra", "Radiolarian",
                    "Vorticella", "Daphnia", "Gonium", "Trichomonas", "Nematode"],
                "Mythical Creatures": [
                    "Dragon", "Unicorn", "Phoenix", "Griffin", "Mermaid",
                    "Centaur", "Minotaur", "Basilisk", "Chimera", "Pegasus",
                    "Gorgon", "Werewolf", "Vampire", "Yeti", "Sphinx",
                    "Kraken", "Hydra", "Leprechaun", "Banshee", "Cerberus"]
            },
            "Organs": {
                "Digestive System Organs": [
                    "Mouth", "Esophagus", "Stomach", "Cardia",
                    "Liver", "Pancreas", "Gallbladder", "Salivary Glands", "Appendix",
                    "Rectum", "Anus", "Duodenum", "Jejunum", "Pharynx",
                    "Ileum", "Cecum", "Sigmoid Colon", "Pylorus"],
                "Respiratory System Organs": [
                    "Lungs", "Trachea", "Bronchi", "Diaphragm", "Nasal Cavity",
                    "Larynx", "Bronchioles", "Alveoli", "Pleura",
                    "Nasopharynx", "Oropharynx", "Laryngopharynx", "Sinuses", "Epiglottis",
                    "Vocal Cords", "Carina", "Pleural Cavity", "Intercostal Muscles", "Respiratory Mucosa"],
                "Circulatory System Organs": [
                    "Heart", "Arteries", "Veins", "Capillaries", "Aorta",
                    "Vena Cava", "Pulmonary Arteries", "Pulmonary Veins", "Coronary Arteries", "Saphenous Vein",
                    "Arterioles", "Venules", "Endothelium", "Pericardium", "Mitral Valve",
                    "Aortic Valve", "Tricuspid Valve", "Pulmonary Valve", "Cardiac Muscle", "Chambers of the Heart"],
                "Nervous System Organs": [
                    "Spinal Cord", "Cranial Nerves", "Spinal Nerves", "Autonomic Nerves", "Cerebellum",
                    "Medulla Oblongata", "Pons", "Hypothalamus", "Thalamus",
                    "Ganglia", "Amygdala", "Hippocampus", "Frontal Lobe",
                    "Parietal Lobe", "Temporal Lobe", "Occipital Lobe"],
                "Locomotor System Organs": [
                    "Cartilage", "Femur", "Humerus", "Biceps", "Triceps",
                    "Quadriceps", "Hamstrings", "Deltoid", "Elbow Joint", "Knee Joint",
                    "Ankle Joint", "Wrist Joint", "Shoulder Joint",
                    "Achilles Tendon", "Rotator Cuff", "Spine", "Pelvis", "Patella"]
            },
            "Cells": {
                "Epithelial Cells": [
                    "Squamous Epithelial Cells", "Columnar Epithelial Cells", "Cuboidal Epithelial Cells",
                    "Ciliated Epithelial Cells", "Basal Cells", "Glandular Cells", "Transitional Epithelial Cells",
                    "Keratinocytes", "Mucous Cells", "Serous Cells", "Endothelial Cells", "Mesothelial Cells",
                    "Microvilli Cells", "Hair Follicle Cells", "Intestinal Absorptive Cells", "Alveolar Epithelial Cells",
                    "Thyroid Follicular Cells", "Gastric Parietal Cells", "Renal Tubular Cells", "Hepatocytes"],
                "Muscle Cells": [
                    "Skeletal Muscle Cells", "Cardiac Muscle Cells", "Smooth Muscle Cells",
                    "Myocytes", "Myoblasts", "Satellite Cells", "Purkinje Fibers", "Intercalated Discs",
                    "Vascular Smooth Muscle Cells", "Gastrointestinal Smooth Muscle Cells", "Uterine Muscle Cells",
                    "Diaphragm Muscle Cells", "Myoepithelial Cells", "Pericytes", "Muscle Spindle Cells",
                    "Extraocular Muscle Cells", "Papillary Muscle Cells", "Sphincter Muscle Cells",
                    "Arrector Pili Muscle Cells", "Tongue Muscle Cells"],
                "Nerve Cells": [
                    "Neurons", "Sensory Neurons", "Motor Neurons", "Interneurons",
                    "Pyramidal Cells", "Purkinje Cells", "Schwann Cells", "Oligodendrocytes",
                    "Astrocytes", "Microglia", "Ganglion Cells", "Spinal Cord Neurons", "Cerebral Cortex Neurons",
                    "Hippocampal Neurons", "Thalamic Neurons", "Hypothalamic Neurons", "Cerebellar Neurons",
                    "Autonomic Ganglia Neurons", "Photoreceptor Cells", "Bipolar Cells"],
                "Blood Cells": [
                    "Red Blood Cells", "Platelets", "Lymphocytes",
                    "Neutrophils", "Eosinophils", "Basophils", "Monocytes", "Macrophages",
                    "B Cells", "T Cells", "Natural Killer Cells", "Dendritic Cells",
                    "Plasma Cells", "Megakaryocytes", "Erythroblasts", "Myeloblasts", "Proerythroblasts",
                    "Hemocytoblasts"],
                "Connective Tissue Cells": [
                    "Fibroblasts", "Chondrocytes", "Osteoblasts", "Osteocytes",
                    "Osteoclasts", "Adipocytes", "Mast Cells", "Mesenchymal Cells", "Tenocytes",
                    "Ligament Cells", "Tendon Cells", "Stromal Cells", "Keratinocytes", "Melanocytes",
                    "Endothelial Cells of Blood Vessels", "Pericytes of Blood Vessels"]
            },
        }

    def __call__(self, a, idx, fix_categ=None) -> list:
        if not a:
            # generate sequences
            if fix_categ is None:
                categ_list = random.choice(self.categ_list)
            else:
                categ_list = self.categ_list[fix_categ]
            #print(f"cate idx = {idx}, categ_list = {categ_list}")
            #categ_list = self.categ_list[2]
            #return categ_list[0:idx]
            choices = range(len(categ_list) - idx + 1)
            choice = random.choice(choices)
            return categ_list[choice : choice+idx]
        else:
            # generate items and put it into self.unique if it is unique
            item_pool = self.categ_dict[a]
            #print(item_pool)
            #print(idx)
            item_pool = random.choice(list(item_pool.values()))
            return random.sample(item_pool, idx)

    def self_check(self):
        '''
        check if the categ is valid. Otherwise print the repeated items.
        '''
        for categ_seq in self.categ_list:
            item_set = set()
            for categ in categ_seq:
                if categ not in item_set:
                    item_set.add(categ)
                else:
                    print("Categ", categ)
                    return
                item_dict = self.categ_dict[categ]
                for key, val in item_dict.items():
                    if key not in item_set:
                        item_set.add(key)
                    else:
                        print("Key", key)
                        return
                    for item in val:
                        if item not in item_set:
                            item_set.add(item)
                        else:
                            print("Item", item)
                            return
        print("Valid")
        return