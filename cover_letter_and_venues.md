# Cover letter (draft) and venue shortlist

## Cover letter

Dear Editor,

We submit "How Much Do Coordination Graphs Add to Opinion-Spam Detection Beyond Reviewer Context?" for consideration in [JOURNAL].

Graph methods for opinion spam are usually reported as improvements over text-only baselines. We ask a narrower question: how much of the graph gain remains once each reviewer's own account-level score is given to the model? We build a multiplex coordination graph (synchrony, content, media layers) with a degree-corrected Poisson null and FDR control, and evaluate it under a nested, reviewer-grouped protocol on YelpZip and YelpNYC, plus injected-campaign benchmarks.

Main findings, all reported with their limits:
1. On YelpZip the graph adds about +0.013 AP over account context (0.465 to 0.478), and a trained GraphSAGE and a no-graph MLP are within noise of the proposed scorer.
2. For label-propagation-style graph additions, 92-97% of the apparent gain disappears once account context is controlled; for SIGN-style features, 25-27% does.
3. On injected campaigns the validated graph helps substantially (delta AP 0.12-0.20), except against the strongest adversary scenario (0.009).
4. On a platform-recommended Yelp Open sample, 25.5% of validated synchrony edges join Yelp friends (about 93 times chance), so friend groups are a false-alarm source.

We do not claim a new state of the art. The contribution is the control, the label-free edge-validation rule, and a documented account of when graph structure adds signal. All data are public (Yelp datasets, used under their terms); code and results are in [REPOSITORY DOI]. [AI-USE STATEMENT: state truthfully the role of AI tools in analysis and writing.] The work has not been published elsewhere and is not under review elsewhere. A previous version was submitted to Social Network Analysis and Mining and was not sent for review (desk rejection); this version has new experiments and a changed framing. [Confirm this with the authors and say so openly.]

Sincerely,
[Corresponding author], on behalf of all authors

## Venue shortlist (verify scope, quartile and APC on SCImago/journal site before submitting)

| Venue | Why it may fit | Risk |
|---|---|---|
| Information Processing & Management | Publishes review-spam and fake-review work; accepts evaluation-focused papers | Wants clear novelty; long review |
| Knowledge-Based Systems | Graph-based fraud detection is in scope | Expects methodological novelty and stronger gains |
| IEEE Trans. on Computational Social Systems | Social-network analysis fit | Review time varies |
| Decision Support Systems | Fake-review detection and platform decisions | Expects managerial framing |
| Expert Systems with Applications | Fast, broad scope | Quartile and reputation vary; reviewers push for bigger benchmarks |
| Online Social Networks and Media / Social Network Analysis and Mining | Fits the topic | SNAM already desk-rejected; ask why first |

Recommendation: first target IP&M, with TCSS as backup. Frame the paper as a controlled evaluation, not a new detector. Ask the SNAM editor why it was desk-rejected and answer that in this cover letter.

## Still needed from the authors
Repository DOI, author contributions, ethics and data-use statement, funding, truthful AI-use statement, final author list (Niranjan Rathore is not on it).
