# Result data for "How Much Do Coordination Graphs Add to Opinion-Spam Detection Beyond Reviewer Context?"

These CSV files contain the derived results behind every table and figure of the paper. They contain **no raw Yelp reviews, no reviewer identifiers and no friendship lists**. YelpZip and YelpNYC (Rayana and Akoglu, 2015) and the Yelp Open Dataset (Yelp Inc.) must be obtained from their owners under their own terms.

Metric columns: ap = average precision, auroc = area under the ROC curve, f1 = F1 at the validation-chosen threshold, rec_fpr1 = recall at 1% false-positive rate, ece = expected calibration error, brier = Brier score.

| File | Content |
|---|---|
| d01_yelpzip_main_results_per_seed.csv | YelpZip, review level, nested protocol, 3 seeds: text only, stage fusion, account context, MultiSentinel-G (k_min=2), GraphSAGE and no-graph MLP (with/without the account-context column), plus ablation variants (method column) |
| d02_yelpnyc_main_results_per_seed.csv | Same for YelpNYC |
| d03_account_context_control_yelpzip.csv | AP gain of label-propagation and SIGN-style graph additions without and with the account-context control |
| d04_kmin_permutation_diagnostic.csv | Validated near-duplicate edges on real and authorship-permuted data, by k_min (YelpZip, YelpNYC) |
| d05_parameter_sensitivity_yelpzip.csv | Sync window, FDR level and propagation weight sweeps, 3 seeds |
| d06_injection_yelpzip_background.csv | Injected campaigns (six scenarios, 3 seeds, prevalence sweep) on the genuine-account YelpZip background |
| d07_injection_yelp_open_background.csv | Same on the Yelp Open background, with friend edges kept or removed |
| d08_friend_group_test_yelp_open.csv | Share of validated edges that join Yelp friends, versus chance |
| d09_cooccurrence_friend_baseline_yelp_open.csv | Friend rate among pairs reviewing the same restaurant in the same week versus other weeks |
| d10_campaignsim_simulation_results.csv | Results of the fully synthetic CampaignSim simulation (nine scenarios, many seeds) |

Variant column in d06/d07: main, prevalence_sweep, friends_kept, friends_removed. Scenario names I0-I5 are injection scenarios; S0-S8 are simulation scenarios.

Licence: CC BY 4.0 for these derived result files.
