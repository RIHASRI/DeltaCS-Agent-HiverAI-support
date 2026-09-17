import re
from typing import Dict, Any, Tuple, Optional

class DeltaResponseGenerator:
    """
    Brand Reply Generator for Delta Air Lines customer support (@Delta).
    Generates intent-conditioned, context-aware, tone-aligned customer service replies.
    """
    
    BRAND_VOICE = {
        "greeting": "Hello!",
        "empathy_phrases": {
            "frustrated": "We sincerely apologize for the distress and frustration this situation has caused.",
            "negative": "We apologize for the inconvenience you experienced with your travel plans.",
            "neutral": "Thanks for reaching out to Delta Air Lines customer support.",
            "positive": "Thank you for reaching out to Delta! We're glad to assist you."
        },
        "links": {
            "rebook": "https://www.delta.com/rebook",
            "baggage": "https://www.delta.com/bagtrac",
            "refunds": "https://www.delta.com/refunds",
            "checkin": "https://www.delta.com/checkin",
            "general": "https://www.delta.com"
        }
    }

    def extract_context_entities(self, query_text: str) -> Dict[str, Optional[str]]:
        """Extract key entities from tweet text: flight number, baggage tag, pnr/confirmation code."""
        entities = {}
        
        # Flight Number (e.g. DL1421, DL 502)
        flight_match = re.search(r'\b(DL\s?\d{1,4})\b', query_text, re.IGNORECASE)
        entities['flight_number'] = flight_match.group(1).upper().replace(" ", "") if flight_match else None
        
        # Baggage Tag / Claim Ref (e.g. #DL983211, #BG99201, Tag DL983211)
        bag_match = re.search(r'#(?:DL|BG)?\d{5,8}\b|\bTag\s*#?[A-Z0-9]{5,8}\b', query_text, re.IGNORECASE)
        entities['baggage_tag'] = bag_match.group(0).upper() if bag_match else None
        
        # Confirmation Code / PNR (e.g. #H92K1L, code K8912P)
        pnr_match = re.search(r'#[A-Z0-9]{6}\b|\bcode\s*#?[A-Z0-9]{6}\b', query_text, re.IGNORECASE)
        entities['pnr'] = pnr_match.group(0).upper() if pnr_match else None
        
        return entities

    def generate_reply(
        self,
        query: str,
        intent: str,
        sentiment: str = "neutral",
        escalation_status: str = "AUTO_REPLY",
        escalation_reason: Optional[str] = None
    ) -> str:
        """
        Generate a brand-aligned, intent-conditioned response.
        If escalation_status is HUMAN_ESCALATION, generates a secure handoff request.
        """
        entities = self.extract_context_entities(query)
        empathy = self.BRAND_VOICE["empathy_phrases"].get(sentiment, self.BRAND_VOICE["empathy_phrases"]["neutral"])
        
        flight_str = f" regarding {entities['flight_number']}" if entities['flight_number'] else ""
        
        # Escalation Handoff Response
        if escalation_status == "HUMAN_ESCALATION":
            return (
                f"{self.BRAND_VOICE['greeting']} {empathy} "
                f"We want to ensure your request{flight_str} is handled with top priority. "
                "Please DM us your 6-character confirmation code, full name, and phone number so a Delta Customer Care specialist "
                "can assist you directly."
            )

        # Automated Intent-Conditioned Responses
        if intent == "flight_delay_cancellation":
            action_text = (
                f"You can check real-time updates and self-service rebooking options for your flight{flight_str} "
                f"via the Fly Delta app or directly at {self.BRAND_VOICE['links']['rebook']}."
            )
        elif intent == "baggage_issue":
            bag_str = f" (Tag: {entities['baggage_tag']})" if entities['baggage_tag'] else ""
            action_text = (
                f"You can track your baggage location{bag_str} in real time using the Fly Delta app or submit a claim "
                f"at {self.BRAND_VOICE['links']['baggage']}."
            )
        elif intent == "booking_seat_change":
            action_text = (
                "You can select seats, request upgrades, and manage your itinerary under 'My Trips' on Delta.com "
                "or via the Fly Delta app."
            )
        elif intent == "refund_compensation":
            action_text = (
                f"Refund requests and eCredit status can be submitted and tracked online at {self.BRAND_VOICE['links']['refunds']}. "
                "Eligible eCredits are automatically added to your SkyMiles profile."
            )
        elif intent == "checkin_boarding":
            action_text = (
                "Mobile check-in opens 24 hours before departure. If you encounter app errors or need paper passes, "
                "our airport kiosk and gate agents are ready to assist you."
            )
        else: # general_inquiry
            action_text = (
                f"You can find comprehensive details regarding Delta policies, Sky Club access, and onboard amenities at {self.BRAND_VOICE['links']['general']}."
            )

        return (
            f"{self.BRAND_VOICE['greeting']} {empathy} {action_text} "
            "Please DM us if you have any additional questions! *Delta CS"
        )

if __name__ == "__main__":
    generator = DeltaResponseGenerator()
    sample_query = "My flight DL1421 was delayed and I need help! #DL1421"
    reply = generator.generate_reply(sample_query, "flight_delay_cancellation", sentiment="frustrated", escalation_status="AUTO_REPLY")
    print("Sample Generated Reply:")
    print(reply)
