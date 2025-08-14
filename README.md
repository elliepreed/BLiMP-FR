**BLiMP-FR: French BLiMP**

BLiMP-fr is a French adaptation of the BLiMP (Benchmark of Linguistic Minimal Pairs) suite, designed to evaluate language models’ grammatical knowledge through minimal-pair judgment tasks. Each example consists of two nearly identical French sentences - one grammatically correct, one incorrect - and the model’s task is to identify the correct one. The benchmark covers a range of syntactic and morphological phenomena, allowing for fine-grained analysis of a model’s linguistic competence in French.

Dataset Structure BLiMP-fr contains eight distinct linguistic phenomena, each representing a specific area of French grammar that language models are tested on:

- **Adjective–noun agreement** – Tests agreement in gender and number between adjectives and the nouns they modify.
- **Anaphor agreement** – Evaluates correct agreement between pronouns (anaphors) and their antecedents.
- **Auxiliary agreement** – Checks subject–auxiliary agreement in tense, number, and person.
- **Binding** – Tests binding principles, e.g., correct reference of pronouns and reflexives within syntactic domains.
- **Clitic placement** – Evaluates correct positioning of clitic pronouns in French sentences.
- **Determiner-noun agreement** – Tests appropriate use and agreement of determiners with nouns.
- **Negation** – Checks correct formation of negative constructions in French.
- **Subjunctive** – Evaluates proper use of the subjunctive mood in subordinate clauses.
Each set contains minimal pairs (one grammatical, one ungrammatical) so models can be scored on their ability to select the correct form.

**Uses**

These eight phenomena were selected because many of them are typologically distinct from their English counterparts, enabling the evaluation to probe a model’s ability to adapt and generalize beyond the structures of its primary training language. By focusing on areas where French diverges substantially from English - such as agreement morphology, clitic placement, and the subjunctive mood - the benchmark tests the extent to which a model can expand its grammatical competence by analogy to a typologically different language. This design also allows for the detection of cross-linguistic interference and the assessment of transfer effects, providing insight into how prior knowledge of English influences performance in French.

