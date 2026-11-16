# B5 Explanation Report (lr_v1, SHAP + registry-driven translator)

Explainer: shap.LinearExplainer on the fitted LogisticRegression (the lr_v1
pipeline's clf), operating on StandardScaler-transformed inputs.
Output space: LOG-ODDS. base_value = 2.620683 (decision at the train-set
background mean); base + sum(SHAP) == decision_function exactly
(max residual on val: 1.8e-15, asserted with tolerance 1e-4).
Background: full train split (n=479), deterministic row order, seed 42.
All 19 features retained in the calculation (none dropped - correlated
features share/redistribute attribution, which is documented, not hidden).

## Global importance (VAL split, n=61; test set not used)

| rank | feature | mean abs SHAP | mean signed |
|---|---|---|---|
| 1 | domain_length | 1.9148 | -0.1911 |
| 2 | url_length | 1.3548 | -0.0012 |
| 3 | num_dots | 1.2749 | +0.1173 |
| 4 | has_https | 0.8276 | +0.0258 |
| 5 | num_hyphens | 0.8197 | -0.0949 |
| 6 | num_subdomains | 0.6192 | +0.1669 |
| 7 | num_digits | 0.4885 | -0.0414 |
| 8 | digit_in_domain_ratio | 0.3982 | -0.0054 |

Interpretation: domain_length dominates - consistent with the B3
correlations and the constructed-legit design (short popular domains vs
long phishing-looking domains). url_length and domain_length (0.80
collinear in B3) split attribution, so the "true" length effect is the sum
of both rows; num_dots/num_subdomains likewise share "domain structure".
Signed values show direction: num_dots and num_subdomains carry positive
signed mass overall (more dots/subdomains -> more phishing), while
domain_length's signed mass is negative because long domains in THIS dataset
are dominated by phishing URLs but the model also uses short popular-domain
legits as its "safe" anchor. Correlated features redistribute credit among
themselves - none is unimportant merely because a neighbor absorbed its
share; grouped presentation concepts (sums of signed contributions) are
kept alongside the raw per-feature values.

## Local cases (defanged; probabilities at threshold 0.2994)

1. Val phishing, p=1.0000, base 2.621, decision 16.988
   - toward phishing: has_https +3.72, domain_length +3.50, num_hyphens +2.53, num_digits +1.49
   - away: num_subdomains -0.26, num_url_params -0.08
   - "The URL has no HTTPS encryption, which the model weighs toward phishing."
   - "The domain name is unusually long."
2. Val legitimate, p=0.0054, decision -5.218
   - away from phishing: domain_length -2.54, url_length -1.34, num_dots -1.26, num_hyphens -0.74
   - "The domain name is short, like many familiar sites, which lowers the
      model's phishing score." / "The site uses HTTPS encryption, which the
      model weighs toward legitimate."
   (A low score is presented as reduced phishing signals, never "safe".)
3. Test FALSE NEGATIVE uspostaldirect[.]com, p=0.0732 (below threshold)
   - The model saw few dots, a short domain and few hyphens - the classic
     bare-domain phishing shape the lexical feature set cannot flag. This is
     the documented blind spot; host-based features (deferred) are the fix.
4. Test LR FALSE POSITIVE shopee[.]co[.]id/images/hero.png (constructed
   legit), p=0.6752 (over threshold)
   - driven by num_dots +3.16 (shopee.co.id has 4 dots vs typical 2-3),
     offset by short-domain and https signals that were not enough.
   - Artifact of the constructed-legit design, disclosed in B4.

## Translation examples

toward phishing: "The URL contains many digits, often used to disguise domains."
away from phishing: "The URL has no unusual special characters." (reduced score)
All sentences come from registry phrases; the translator adds a fixed
disclaimer: "These points describe why the model made this prediction; they
are not proof that the URL is phishing or safe."

## Coverage

Top-5 explanation coverage on val (n=61): **100% (61/61 rows, mean 1.000)**
after the B5 registry phrase enrichment (registry v1.1, phrase-only change -
feature names/types/order and the trained model are unchanged; no
fallback sentences were needed). Zero-contribution features are skipped
rather than emitting a direction sentence.

## Tests

tests/test_explain.py: 15/15 passing (additivity, alignment, direction,
determinism, top-k signs, translator determinism/ranking/fallback/coverage,
no fabricated text, binary-feature phrase-direction guard, group coverage).
A passing translation test is NOT evidence the model is accurate.
