# Data Contract

## Dataset

The project uses a synthetic, pseudonymized dataset representing SEO and content performance records. It is created for educational and portfolio use and does not contain real FlyRank or confidential data.

## Feature Window

The feature window contains the columns available at prediction time. These fields describe content metadata, historical performance, content quality, and user engagement.

### Content metadata

- content_id: unique identifier for a content record
- content_type: type of asset such as blog, landing_page, guide, product_page, or news
- category: content topic area
- word_count: total word count
- content_age_days: age of the content in days
- author_type: internal or external author type
- has_schema: whether structured data/schema markup is present
- internal_link_count: number of internal links
- external_link_count: number of external links

### SEO and organic discovery

- organic_clicks: total organic clicks in the observation window
- organic_impressions: total organic impressions in the observation window
- ctr: click-through rate
- average_position: average search ranking position
- keyword_count: number of tracked keywords
- ranking_keywords: number of ranking keywords
- backlinks: number of backlinks
- domain_authority: domain authority score

### Engagement

- sessions: content sessions
- bounce_rate: bounce rate
- avg_session_duration: average session duration in seconds
- conversions: conversions attributed to the content

### Historical performance windows

- clicks_7d / clicks_30d
- impressions_7d / impressions_30d
- position_7d / position_30d

## Label and Future Window

The label and future-derived fields should never be used as model features because they represent information outside the prediction-time feature window.

- trend_direction: direction of movement discovered from the future trend window
- trend_pct: percentage trend in the future window
- future_clicks: clicks in a future period
- future_position: future ranking position

## Target Definition

The target column is:

- is_declining_label

This is a binary label where:

- 1 = declining performance
- 0 = not declining performance

The target is built from future-derived signals such as a future decline in clicks, a worse future search position, or a future trend direction that is down.

## Leakage Risk

These fields must not be used as model features in a prediction-time model:

- trend_direction
- trend_pct
- future_clicks
- future_position

They are label-correlated and directly encode future behavior. Training a model with them creates data leakage.

## Missing Values

The synthetic dataset is generated without missing values. In project use, the preprocessing pipeline should treat missing values using median imputation for numeric features and most-frequent imputation for categorical features.

## Data Types

The dataset is mostly numeric and categorical. Categorical fields should be encoded using one-hot encoding or similar methods. Numeric fields should be scaled where a model requires it.

## Modeling Rule

The project must enforce a temporal feature separation:

- Training data uses observations from an earlier time period
- Validation and test data use later time periods

This avoids random data leaks and keeps the intended modeling story aligned with time-aware prediction.
