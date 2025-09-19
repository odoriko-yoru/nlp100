# Chapter6

## gensimパッケージの依存関係

2025-06-01時点でののgensim(4.3.3)はnumpy2系へ対応できていない。

そのため、numpyは1系へダウングレードした。

## No.55の意味的アナロジーと文法的アナロジー

`questions-words.txt`の出典は[Distributed Representations of Words and Phrases and their Compositionality](https://doi.org/10.48550/arXiv.1310.4546)である。論文には意味的アナロジー(semantic analogy)と文法的アナロジー(syntactic analogy)についての記載がある(3.Empirical Results)。

>The task consists of analogies such as “Germany” : “Berlin” :: “France” : ?, which are solved by finding a vector x such that vec(x) is closest to vec(“Berlin”) - vec(“Germany”)+ vec(“France”) according to the cosine distance (we discard the input words from the search). Thisspecific example is considered to have been answered correctly if x is “Paris”. The task has two broad categories: the syntactic analogies (such as “quick” : “quickly” :: “slow” : “slowly”) and the semantic analogies, such as the country to capital city relationship.

### 意味的アナロジー

意味的アナロジーは単語同士の意味関係の類推を指す。文中では以下のうち「?」を類推することとしている。

“Germany” : “Berlin” :: “France” : ?

もしWord2Vecが"Germany", "Berlin", "France"から"Paris"を類推することができれば、この問題は「正答」となる。

### 文法的アナロジー

一方で文法的アナロジーとは単語のPOSの類推と考えて良い。文中の例は以下である。

“quick” : “quickly” :: “slow” : “slowly”

具体例では「形容詞 : 副詞」の関係を類推することが期待されている。

## No.55はどう解くか

`questions-words.txt`には複数のセクションがあり意味を類推するタスク(=意味的アナロジー)と文法関係を類推するタスク(=文法的アナロジー)に分かれている。

1. capital-common-countries
1. capital-world
1. currency
1. city-in-state
1. family
1. gram1-adjective-to-adverb
1. gram2-opposite
1. gram3-comparative
1. gram4-superlative
1. gram5-present-participle
1. gram6-nationality-adjective
1. gram7-past-tense
1. gram8-plural
1. gram9-plural-verbs

1-5は意味的アナロジー、6-14は文法的アナロジーのタスクであり、学習済みWord2Vecで1-3列目の単語情報を基に4列目の単語を類推する。単語が一致していれば正答となる。

No.55は`capital-common-countries`に対して意味的な類推を行い、4列目の単語(Ground Truth)と類推された単語の一致率を算出することが期待されている。したがって、No.55のタスクについては文法的アナロジーの正答率の算出は必要ない。
