"""Feature taxonomy helpers for the real SEO dataset."""

from __future__ import annotations

FEATURE_GROUPS = {
    "Temporal": ["Date", "Month", "Year", "Quarter", "Time Of Day"],
    "Search_SEO": [
        "Primary Keywords",
        "Secondary Keywords",
        "Long-Tail Keywords",
        "Keywords_Ranking",
        "Clicks",
        "Impressions",
        "CTR (%)",
        "Average_Position",
        "Backlinks",
        "Domain_Authority",
        "Keyword_Difficulty",
        "Indexed_Pages",
    ],
    "Traffic": [
        "Organic_Traffic",
        "Referral_Traffic",
        "Organic_vs_Paid_Traffic_Split (%)",
        "Mobile_vs_Desktop_Traffic_Split (%)",
    ],
    "Engagement": [
        "Bounce_Rate (%)",
        "Pages_per_Session",
        "Average_Session_Duration (sec)",
        "New_vs_Returning_Visitors (%)",
        "Exit_Rate (%)",
    ],
    "Business_Conversion": [
        "Conversion_Rate (%)",
        "Goal_Completions",
        "Organic_Revenue ($)",
    ],
    "Technical_SEO": [
        "Page_Load_Time (sec)",
        "Core_Web_Vitals_LCP (sec)",
        "Core_Web_Vitals_FID (ms)",
        "Core_Web_Vitals_CLS",
        "Technical_SEO_Errors",
        "Time_to_First_Byte_TTFB (ms)",
    ],
    "Contextual": [
        "Location",
        "Social Media Source",
        "Media Type",
        "Device Type",
        "Top_Landing_Pages",
    ],
}


def get_feature_groups() -> dict[str, list[str]]:
    """Return the documented SEO feature taxonomy."""
    return FEATURE_GROUPS.copy()
