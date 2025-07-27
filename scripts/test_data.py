import random
from typing import List, Tuple


def get_test_samples() -> List[Tuple[str, str | None]]:
    """Get test samples with conversation examples"""
    return [
        ("Perfect.", "Great, I'll update the schedule right now."),
        ("oh", "I'm calling regarding the recent shipment documentation."),
        ("Absolutely.", "Perfect, I'll do that now."),
        ("I see.", "We actually received the PO on our end."),
        ("Makes sense.", "Great question. We're looking at streamlining."),
        ("Okay", "Got it. That can happen if your email was used previously."),
        ("mhm", "Exactly. Dallas has seen a 40% increase."),
        ("Thanks so much for your help.", "Sure, I'll add that note to your order."),
        ("Yes, please send", "I can email you the updated SDS right away."),
        ("Perfect, thanks again.", "No problem at all. I'll call you back soon."),
        (
            "Right. There's more",
            "Got it. Once inside, should they unload left or right?",
        ),
        ("Yes, this", "Good morning, this is Alex from Global Imports."),
        (
            "Great, thank you. Let me pull up the details",
            "Sure, the order number is 45582.",
        ),
        ("uh huh", None),
        ("yeah", None),
        ("mm hmm", None),
        ("okay", None),
        ("right", None),
        ("I think we should go to the store", None),
        ("What time is the meeting tomorrow?", None),
        ("The weather is really nice today", None),
        ("Can you help me with this problem?", None),
        ("I don't understand what you mean", None),
        ("sure", None),
        ("maybe", None),
        ("I see", None),
        ("interesting", None),
        ("yeah", "What do you think about the new policy?"),
        ("uh huh", "The meeting is scheduled for 3 PM"),
        ("right", "We need to finish this by Friday"),
    ]


def generate_test_data(num_samples: int) -> List[Tuple[str, str | None]]:
    """Generate test data for stress testing"""
    test_examples = get_test_samples()

    test_data = []
    for _ in range(num_samples):
        text, context = random.choice(test_examples)
        test_data.append((text, context))

    return test_data
