# Interview Notes

## 1. What problem does the project solve?

The project predicts whether an SEO/content record is likely to be declining in performance using a synthetic dataset. It demonstrates how a junior ML workflow can detect and report the risk of data leakage.

## 2. Why was classification used?

The target is binary. A record is either declining or not declining, so classification is the right supervised learning problem.

## 3. What is data leakage?

Data leakage occurs when a model accidentally uses information that would not be available at prediction time, such as future metrics or label-derived fields.

## 4. How did leakage occur?

The dataset includes fields such as trend_direction, trend_pct, future_clicks, and future_position. Those values are generated from later performance windows and are related to the target definition.

## 5. How was leakage detected?

By comparing a leaky model where future-derived fields were included with a leakage-free model where those fields were removed. The leaky experiment usually produces very high performance and suspiciously strong feature importance.

## 6. How was leakage fixed?

The final model uses only features that are available at prediction time and explicitly excludes future-derived and target-correlated fields.

## 7. Why is temporal splitting important?

The data represents an evolving SEO environment. Random splits can mix future patterns into training data and provide an unrealistic evaluation. Time-aware splits maintain a realistic chronology.

## 8. Why is accuracy not enough?

Accuracy can be misleading when the class distribution is imbalanced. Here, precision, recall, F1-score, ROC-AUC, PR-AUC, and Precision@50 are more useful because they explain how well the classifier finds the declining records.

## 9. What is Precision@50?

Precision@50 measures whether the top 50 most likely declining records contain a useful rate of true declining records. It is useful for ranking and prioritization.

## 10. Why was Random Forest used?

Random Forest is a useful baseline because it can capture nonlinear relationships and automatically provide feature importance information without heavy tuning.

## 11. What are the limitations?

The dataset is synthetic and simplified. The project is not a real production deployment pipeline and does not include live SEO monitoring or a production ML platform.

## 12. What would be done with real company data?

I would work with a legal and privacy-safe data contract, define a prediction-time feature window, and run a leakage audit before training a model.

## 13. How would you deploy the model?

I would package the preprocessing and model as a Python service, expose predictions through an API, and connect it to a monitoring dashboard.

## 14. How would you monitor the model?

I would log prediction inputs, outputs, drift, class distribution, and feature distribution, and I would refresh the model when the data contract or target changes.

## 15. How would you prevent leakage in a production ML pipeline?

I would clearly separate the feature window from the label/future window, enforce schema validation, create a data contract, review feature sources, and require a release checklist that checks time-order dependencies.
