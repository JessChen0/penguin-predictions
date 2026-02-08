# penguin_predictions_2026


## Introduction

This project leverages the penguin_predictions.csv dataset (below) to
examine classification problems and metrics.

``` r
library(tidyverse)
```

    Warning: package 'ggplot2' was built under R version 4.5.2

    ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
    ✔ dplyr     1.1.4     ✔ readr     2.1.5
    ✔ forcats   1.0.0     ✔ stringr   1.5.1
    ✔ ggplot2   4.0.1     ✔ tibble    3.3.0
    ✔ lubridate 1.9.4     ✔ tidyr     1.3.1
    ✔ purrr     1.1.0     
    ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ✖ dplyr::filter() masks stats::filter()
    ✖ dplyr::lag()    masks stats::lag()
    ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors

``` r
library(knitr)
library(caret)
```

    Loading required package: lattice

    Attaching package: 'caret'

    The following object is masked from 'package:purrr':

        lift

``` r
penguin_predictions <- read.csv("https://raw.githubusercontent.com/acatlin/data/refs/heads/master/penguin_predictions.csv")

glimpse(penguin_predictions)
```

    Rows: 93
    Columns: 3
    $ .pred_female <dbl> 0.99217462, 0.95423945, 0.98473504, 0.18702056, 0.9947012…
    $ .pred_class  <chr> "female", "female", "female", "male", "female", "female",…
    $ sex          <chr> "female", "female", "female", "female", "female", "female…

``` r
ggplot(data=penguin_predictions, aes(x=sex, fill=sex)) + geom_bar() +
  labs (title = "Count of actual pengiun sexes.")
```

![](penguin_predictions_2026_files/figure-commonmark/unnamed-chunk-1-1.png)

## Analysis

### Null Error Rate

The null error rate is calculated as 1 - the majority class error rate.
This is important to know as in cases where there is a high class
imbalance, (ie, when one outcome is signficantly high) the model can
show a high percentage accuracy but still be ineffective. In this case,
the null error rate is 0.58 means the data is relatively balanced.

``` r
null_accuracy <- mean(penguin_predictions$sex == "female")
null_error_rate <- 1 - null_accuracy
cat("Null Error Rate is:",null_error_rate) 
```

    Null Error Rate is: 0.5806452

### Confusion Matrices & Performance Metrics

We calculate confusion matrices at three different thresholds to review
changes in metrics as thresholds increase/decrease. We will use
thresholds: 1. 0.2 (pred_0.2) 2. 0.5 (we will use the base dataset,
.pred_class) 3. 0.8 (pred_0.8)

#### 0.2 Threshold

``` r
penguin_predictions <- mutate(penguin_predictions, pred_0.2=ifelse(.pred_female>0.2,"female","male") )

results_0.2 <- confusionMatrix(as.factor(penguin_predictions$pred_0.2), as.factor(penguin_predictions$sex), positive= "female")

results_0.2$table
```

              Reference
    Prediction female male
        female     37    6
        male        2   48

``` r
#Metrics
results_0.2$overall["Accuracy"] #TP+TN / Total
```

     Accuracy 
    0.9139785 

``` r
results_0.2$byClass["Precision"] #TP / (TP+FP)
```

    Precision 
    0.8604651 

``` r
results_0.2$byClass["Recall"]  #TP / (TP + FN)
```

       Recall 
    0.9487179 

``` r
results_0.2$byClass["F1"]
```

          F1 
    0.902439 

#### 0.5 Threshold

``` r
results_0.5 <- confusionMatrix(as.factor(penguin_predictions$.pred_class), as.factor(penguin_predictions$sex), positive= "female")

results_0.5$table
```

              Reference
    Prediction female male
        female     36    3
        male        3   51

``` r
#Metrics
results_0.5$overall["Accuracy"] #TP+TN / Total
```

     Accuracy 
    0.9354839 

``` r
results_0.5$byClass["Precision"] #TP / (TP+FP)
```

    Precision 
    0.9230769 

``` r
results_0.5$byClass["Recall"]  #TP / (TP + FN)
```

       Recall 
    0.9230769 

``` r
results_0.5$byClass["F1"]
```

           F1 
    0.9230769 

#### 0.8 Threshold

``` r
penguin_predictions <- mutate(penguin_predictions, pred_0.8=ifelse(.pred_female>0.8,"female","male") )

results_0.8 <- confusionMatrix(as.factor(penguin_predictions$pred_0.8), as.factor(penguin_predictions$sex), positive= "female")

results_0.8$table
```

              Reference
    Prediction female male
        female     36    2
        male        3   52

``` r
#Metrics
results_0.8$overall["Accuracy"] #TP+TN / Total
```

     Accuracy 
    0.9462366 

``` r
results_0.8$byClass["Precision"] #TP / (TP+FP)
```

    Precision 
    0.9473684 

``` r
results_0.8$byClass["Recall"]  #TP / (TP + FN)
```

       Recall 
    0.9230769 

``` r
results_0.8$byClass["F1"]
```

           F1 
    0.9350649 

## Discussion and threshold Use Cases

In the above analysis, we analyzed various confusion matrices and
metrics at varying thresholds. In this example, where the classes are
fairly balanced, the resulting metrics (using F1 as a harmonic mean
between precision and recall) increase subtly as the threshold
increases.

In practical application, there may be cases where a lower threshold
(more positives predicted) would be preferred. For example, in cases of
disease detection, we would be more interested in diagnosing everyone
who has the disease, which could lead to more false positives.

On the other hand, in spam detection, where spam is assigned a positive
value, we would not want to mark any important emails as spam.
Therefore, we would want to minimize false positives so that we only put
spam emails directly into trash.

## Sources:

Google DeepMind. (2025). Gemini 3 Flash \[Large language model\].
https://gemini.google.com. Accessed Feb 7, 2026 (chat link:
https://gemini.google.com/share/76406da30089)

Sunasra, Mohammed. (Nov 11, 2017). “Performance Metrics for
Classification problems in Machine Learning”.
https://github.com/acatlin/data/blob/master/Performance%20Metrics%20for%20Classification%20problems%20in%20Machine%20Learning.pdf.
