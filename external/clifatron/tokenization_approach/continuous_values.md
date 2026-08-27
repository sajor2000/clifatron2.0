# Tokenization strategy for Continuous values in CLIF

0. use published reference material to define the Lower limit of normal (LLN) and Upper Limit of Normal (ULN) for each `*_category` continuous variable
1. Create quantiles with the normal range (`<category>_normal_1` ,.....`<category>_normal_10`)
2. for abnormal values, three different strategies depending on clinical domain knowledge
* `low` = create `<category>_low_1 ,.....<category>_low_10` and a single token `<category>_high`
* `high` = `create <category>_high_1 ,.....<category>_high_10` and a single token `<category>_low`
* `low_high` = create both the low and high quantiles for values above and below the normal range 


notes: if the LLN = zero, there should be no `low` token.
