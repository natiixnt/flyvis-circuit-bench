<h1 align="center">flyvis-circuit-bench</h1>

<p align="center"><b>Ile z tego, na czym opiera się obwód neuronowy, da się odczytać ze schematu połączeń?
Benchmark metod odkrywania obwodów względem dokładnej prawdy z ablacji w modelu układu wzrokowego muchy
ograniczonym konektomem.</b></p>

<p align="center">
  <a href="../../actions/workflows/ci.yml"><img alt="CI: GitHub Actions" src="https://github.com/natiixnt/flyvis-circuit-bench/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="Kod: MIT" src="https://img.shields.io/badge/code-MIT-blue"></a>
  <a href="LICENSE-data"><img alt="Dane: CC BY 4.0" src="https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey"></a>
  <img alt="Python od 3.11 do 3.13" src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue">
  <a href="https://colab.research.google.com/github/natiixnt/flyvis-circuit-bench/blob/main/notebooks/score_your_method.ipynb"><img alt="Otwórz w Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/figures/hero-dark.svg">
    <img alt="Udział osiągalnego sygnału, sieci wytrenowane, 10 modeli, kolor według tego, czego metoda
    potrzebuje: sam konektom (wielkość klasy 20%, liczba synaps 53%), wytrenowane wagi (suma wag 76%), przejście
    w przód bez wyuczonych wag (sama aktywność źródła 43%, liczba synaps razy aktywność źródła 67%), wytrenowane
    wagi i przejście w przód (|wagi| razy kod SAE źródła 75%, |wagi| razy aktywność źródła 87%, |wagi| razy
    aktywność neuronu 88%), przejście wstecz (gradient 92%, patching 97%)" src="docs/figures/hero-light.svg" width="520">
  </picture>
</p>

English: [README.md](README.md). Liczby zapisujemy z kropką dziesiętną, jak w wersji angielskiej, w tabelach
(generowanych z wyników, po angielsku) i na wykresach.

## W skrócie

Wyłączaliśmy po kolei każdą z 604 klas połączeń typ komórki do typu komórki w 10 wytrenowanych modelach układu
wzrokowego muchy ([flyvis](https://github.com/TuragaLab/flyvis), Lappalainen i in., Nature 2024) i mierzyliśmy,
jak bardzo zmienia się odpowiedź sieci. To jest prawda. Attribution patching, który szacuje ten sam efekt z
gradientów, jest punktem odniesienia: odzyskuje **97%** osiągalnego sygnału (AUROC 0.987). Pytanie brzmi, ile
daje sama anatomia.

- **Sama anatomia odzyskuje mniej więcej połowę.** Ranking połączeń według liczby synaps odzyskuje 53%
  osiągalnego sygnału (AUROC 0.763); liczba synaps umieszcza 17 z 61 najważniejszych połączeń we własnych
  górnych 61 (precision@61, metryka dodana po prerejestracji).
- **Jeden wyuczony skalar na klasę połączeń domyka mniej więcej połowę pozostałej luki.** We flyvis waga to
  liczba synaps razy jedna wytrenowana siła na klasę; ranking według sumy |wag| odzyskuje 76% (AUROC 0.878),
  czyli 49% luki, którą zostawia liczba synaps.
- **Jedno przejście wstecz daje prawie wszystko.** Pojedynczy gradient osiąga 92% (AUROC 0.961), wobec 97% dla
  patchingu z 32 przejściami wstecz.
- **Sama aktywność, bez wyuczonych wag, niewiele daje.** Średnia |aktywność| typu źródłowego, bez żadnych wag,
  odzyskuje 43% (AUROC 0.715), mniej niż liczba synaps. Liczba synaps razy ta aktywność odzyskuje 67% (AUROC
  0.833), czyli więcej niż 53%, ale mniej niż suma |wag| (76%). To aktywność wytrenowanej sieci na odłożonych
  bodźcach, nie nagranie z muchy; oba wiersze dodano po prerejestracji.
- **Ten SAE nie bije zwykłej sumy wag.** Cechy jedynego rzadkiego autoenkodera, którego dopasowaliśmy, osiągają
  AUROC 0.873, zwykła suma |wag| 0.878.
- **Obalone: trening zmniejsza przewagę patchingu nad anatomią, a nie ją zwiększa.** Na Spearmanie wewnątrz
  górnych 61, jedynej mierze, która nie nasyca się w sieciach niewytrenowanych, przewaga w sieciach
  wytrenowanych minus przewaga w niewytrenowanych wynosi -0.705.

Wszystkie cztery decyzje spisane przed replikacją utrzymały się na 5 modelach analizowanych dopiero potem i na
drugim zestawie bodźców. Masz metodę, która szereguje połączenia? [Oceń ją](#dodaj-swoją-metodę) w kilka minut,
bez GPU, jeśli korzysta z konektomu i wag; metody oparte na aktywacjach i gradientach potrzebują flyvis i torch.

Między modelami efekt jednej klasy połączeń bardzo się zmienia: mediana korelacji Spearmana efektu między dwoma
modelami wynosi 0.33 (v1.1, [docs/v1.1/z3.md](docs/v1.1/z3.md)), więc wnioski dotyczą zespołu modeli, a nie
efektu danej klasy w pojedynczym modelu.

## Dlaczego to ważne

- **Konektomy już są; które połączenia są ważne, wciąż nie wiadomo.** Istnieją pełne konektomy muchy. Ten
  benchmark mierzy, jak daleko sama anatomia pozwala przewidzieć, na których połączeniach opiera się
  wytrenowana sieć, względem prawdy liczonej dokładnie, a nie zgadywanej.
- **Poligon dla metod interpretowalności ze znaną odpowiedzią.** Patching, gradienty, miary oparte na aktywności
  i ten SAE są porównane na tych samych zasadach co anatomia, ze zmierzonym sufitem prawdy, więc każdy wynik da
  się czytać jako udział w tym, co osiągalne.

## Ranking (leaderboard)

Sieci wytrenowane, 10 modeli. Discovery = modele 000-004, analizowane najpierw; replication = modele 005-009,
analizowane po spisaniu reguł. AUROC to prerejestrowana metryka główna; udział sygnału, AUPRC i precision@61
dodaliśmy po prerejestracji i nie zmieniają żadnej decyzji.

| method | needs | AUROC [95% CI] | share of signal | discovery | replication | AUPRC | precision@61 |
|---|---|---|---|---|---|---|---|
| attribution patching | 32 backward passes | 0.987 [0.985, 0.989] | 97% | 0.987 | 0.987 | 0.889 | 0.816 |
| gradient | 1 backward pass | 0.961 [0.955, 0.967] | 92% | 0.961 | 0.961 | 0.804 | 0.715 |
| \|weights\| x neuron activity, per edge | trained weights + forward pass | 0.939 [0.922, 0.955] | 88% | 0.932 | 0.945 | 0.667 | 0.608 |
| \|weights\| x source activity | trained weights + forward pass | 0.936 [0.918, 0.952] | 87% | 0.929 | 0.942 | 0.651 | 0.598 |
| sum of \|weights\| | trained weights | 0.878 [0.855, 0.900] | 76% | 0.879 | 0.878 | 0.414 | 0.428 |
| \|weights\| x source SAE code | trained weights + forward pass + SAE fit | 0.873 [0.831, 0.904] | 75% | 0.859 | 0.888 | 0.488 | 0.474 |
| synapse count x source activity † | connectome + forward pass, no learned weights | 0.833 [0.813, 0.854] | 67% | 0.827 | 0.839 | 0.314 | 0.316 |
| synapse count | connectome only | 0.763 [0.737, 0.787] | 53% | 0.762 | 0.764 | 0.248 | 0.284 |
| source activity only † | forward pass, no learned weights | 0.715 [0.687, 0.744] | 43% | 0.704 | 0.727 | 0.200 | 0.190 |
| class size | connectome only | 0.601 [0.572, 0.633] | 20% | 0.599 | 0.603 | 0.150 | 0.187 |
| truth ceiling | 604 ablations | 1.000 [0.999, 1.000] | 100% | 1.000 | 1.000 | 0.998 | 0.974 |

† dodane po badaniu, poza zarejestrowaną analizą.

AUROC mierzy, jak dobrze ranking 604 klas według metody oddziela górne 10% według efektu ablacji (61 klas).
Udział sygnału (share of signal) = (AUROC - 0.5) / (sufit - 0.5) w każdym modelu: 0% to przypadek, 100% to
zgodność prawdy samej ze sobą między dwiema połowami bodźców. Przewaga patchingu nad liczbą synaps wynosi 0.224, 95% CI [0.201, 0.248]
(bootstrap) i [0.195, 0.253] (t-Studenta), dodatnia w 10 z 10 modeli; na samych modelach discovery było to 0.225
[0.196, 0.256] i [0.175, 0.276]. Wszystkie przedziały, wartości modeli, sieci niewytrenowane i drugi zestaw
bodźców: [docs/results.md](docs/results.md).

**Zgłoszenia.** Otwórz pull request, który dodaje Twój wiersz (wynik `fcb score`) i link do kodu tworzącego
plik z wynikami; [szablon pull requesta](.github/pull_request_template.md) wymienia, co sprawdzamy. Szczegóły w
[CONTRIBUTING.md](CONTRIBUTING.md#add-a-method-to-the-leaderboard).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/figures/per-model-dark.svg">
    <img alt="Liczba synaps kontra attribution patching w każdym z 10 modeli, wytrenowanych i niewytrenowanych:
    w sieciach wytrenowanych patching jest blisko szczytu w każdym modelu, a liczba synaps rozkłada się dużo
    niżej; w niewytrenowanych liczba synaps jest wyżej, a patching leży na suficie"
    src="docs/figures/per-model-light.svg" width="440">
  </picture>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/figures/replication-dark.svg">
    <img alt="Cztery prerejestrowane decyzje replikacji dla wszystkich 10 modeli, modeli discovery i modeli
    replikacji: każdy przedział wyklucza zero w zapowiedzianym kierunku"
    src="docs/figures/replication-light.svg" width="440">
  </picture>
</p>

Anatomia przewiduje sieci niewytrenowane lepiej niż wytrenowane (AUROC liczby synaps 0.865 wobec 0.763), bo
flyvis ustawia na starcie siłę każdej klasy z konektomu (suma wag = 0.01 x liczba krawędzi), więc przed treningiem
każda waga jest stałą funkcją anatomii; trening odsuwa siły od niej.

## v1.1: analizy eksploracyjne †

Dodane po badaniu, zdefiniowane w [docs/analysis-plan-v1.1.md](docs/analysis-plan-v1.1.md) przed uruchomieniem,
oznaczone †; żadna nie zmienia zapisanej decyzji. `fcb reproduce` liczy je od nowa z paczki 1.1.0.

- **Próg ([Z1](docs/v1.1/z1.md)).** Przesunięcie progu z górnych 31 na górne 121 klas nie zmienia kolejności żadnej
  pary metod (0 zamian wśród 45 par); liczba synaps zostaje między AUROC 0.745 a 0.763, patching między 0.982 a
  0.994.
- **Rodziny bodźców ([Z2](docs/v1.1/z2.md)).** Prawda z 24 przesuwanych krawędzi i prawda z 16 klipów Sintel
  zgadzają się (Spearman 0.991); patching ma AUROC 0.986 na krawędziach i 0.987 na klipach, liczba synaps 0.758 i
  0.761.
- **Zgodność między modelami ([Z3](docs/v1.1/z3.md)).** Efekt klasy różni się między modelami: mediana Spearmana
  0.33 między dwoma modelami (1.00 między połówkami bodźców w jednym modelu) i tylko 3 klasy są w górnych 61 co
  najmniej 9 z 10 modeli. Efekt zmierzony w innym modelu przewiduje górne 61 modelu z AUROC 0.727, poniżej liczby
  synaps (0.763), a lepiej od niej w 32 z 90 par modeli.
- **Prawda zadaniowa ([Z4](docs/v1.1/z4.md)).** Efekt ablacji i zmiana straty przepływu optycznego na 16 klipach
  Sintel zgadzają się ze Spearmanem 0.667; z 61 najważniejszych klas wg efektu ablacji 27 jest też w górnych 61 wg
  efektu zadaniowego, a 34 nie.
- **Wyciszanie całego typu ([Z6](docs/v1.1/z6.md)).** Wyciszenie naraz wszystkich klas wychodzących z jednego z 63
  typów źródłowych daje efekt o Spearmanie 0.976 z sumą efektów jego klas; 12 typów maskuje, a 7 działa odwrotnie.
  W 12 z 630 par typ x model wyciszenie rozbija dynamikę sieci, więc te wnioski opieramy na rangach.
- **Ablacja do średniej ([wynik repozytorium badawczego](docs/v1.1/mean-ablation.md)).** Zastąpienie prądu klasy
  jego średnią zamiast zerem zachowuje kolejność metod: patching 0.974 (0.987 dla zerowania), liczba synaps 0.777
  (0.763). Tego wyniku `fcb reproduce` nie liczy od nowa, pomiarów nie ma w paczce.

## Szybki start

Same wyniki, bez GPU, około minuty na laptopie (potrzebny [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/natiixnt/flyvis-circuit-bench && cd flyvis-circuit-bench
uv run fcb download                      # paczka pomiarów 1.1.0, 40.3 MB, sprawdzana sha256
uv run fcb reproduce --data data/measurements-v1.1.0 --check
```

`reproduce` przelicza każdą tabelę w [docs/results.md](docs/results.md) i
[results/results.json](results/results.json) z plików pomiarów i sprawdza zgodność bajt w bajt.
`uv run fcb figures` rysuje wykresy od nowa. Powtórzenie samych pomiarów (`fcb measure`: prawda i każda metoda
z rankingu) wymaga flyvis i torch (`uv sync --extra measure`), zob.
[docs/methodology.md](docs/methodology.md#re-measuring).

## Dodaj swoją metodę

Metoda to cokolwiek, co daje każdej z 604 klas połączeń wynik (większy = ważniejsza), bez patrzenia na ablacje.
Po szybkim starcie:

```python
import numpy as np
from flyvis_circuit_bench import score

DATA = "data/measurements-v1.1.0"
inputs = score.method_inputs(DATA)  # dla modelu: typy komórek, liczby synaps, wagi, krawędzie; bez prawdy

def my_method(x):  # jeden wynik na klasę połączeń
    return x["sum_abs_weight"] / x["n_edges"]  # średnia |waga| na krawędź

np.savez("my_method.npz", **{model: my_method(x) for model, x in inputs.items()})
```

```bash
uv run fcb score --data data/measurements-v1.1.0 --scores my_method.npz --name "mean |weight| per edge"
# scores 0.827 AUROC
```

`fcb score` wypisuje wiersz w formacie [rankingu](#ranking-leaderboard): AUROC z 95% CI po 10 modelach, udział
sygnału, AUROC na modelach discovery i replication, AUPRC i precision@61 (ile z 61 najważniejszych połączeń jest
w Twoich górnych 61). Pola, które daje `method_inputs`, opisuje
[docs/methodology.md](docs/methodology.md#method-inputs). Metody oparte na konektomie i wagach liczą się w kilka
minut bez GPU; metody, które potrzebują aktywacji albo gradientów, ładują modele flyvis po
`uv sync --extra measure` (flyvis i torch). To samo w notatniku:
[notebooks/score_your_method.ipynb](notebooks/score_your_method.ipynb)
([otwórz w Colab](https://colab.research.google.com/github/natiixnt/flyvis-circuit-bench/blob/main/notebooks/score_your_method.ipynb)).

## Jak to działa

1. **Prawda z ablacji.** Dla każdej z 604 klas połączeń (wszystkie synapsy od jednego typu komórki do drugiego)
   ustawiamy jej skalę synaptyczną na 0 i zapisujemy odpowiedź r centralnej kolumny każdego typu komórki na 40
   bodźców (24 przesuwające się krawędzie, 16 klipów z filmu Sintel). Efekt to
   E(e) = średnia po bodźcach z ||r_e - r||² / ||r||². Prawda binarna to górne 10% klas (61).
2. **Metody szeregują klasy bez wglądu w E.** Liczba synaps, suma |wag| i wielkość klasy korzystają tylko z
   konektomu i parametrów. Attribution patching to przybliżenie pierwszego rzędu tego samego efektu z 32 losowych
   kierunków wyjścia (32 przejścia wstecz zamiast 604 ablacji); gradient używa jednego kierunku.
   |Wagi| razy aktywność neuronu (|waga| każdej krawędzi razy średnia aktywność jej neuronu źródłowego), |wagi|
   razy aktywność źródła (suma |wag| klasy razy średnia |aktywność| typu źródłowego) i |wagi| razy kod SAE
   źródła liczymy z aktywności wytrenowanej sieci na odłożonych bodźcach, z wyuczonymi wagami. Sama aktywność
   źródła i liczba synaps razy aktywność źródła używają tej samej aktywności i żadnych wyuczonych wag; dodano
   je po prerejestracji.
3. **Sieci.** 10 wytrenowanych modeli z opublikowanego zespołu flyvis i te same sieci po inicjalizacji flyvis
   (niewytrenowane). Miary: AUROC (główna), AUPRC, Spearman oraz Spearman wewnątrz górnych 61, który nie nasyca
   się, gdy AUROC wynosi 1.
4. **Statystyka.** Średnia po modelach z dwoma 95% przedziałami (bootstrap po modelach i t-Studenta); wynik
   liczy się tylko wtedy, gdy oba się zgadzają. Szczegóły w [docs/methodology.md](docs/methodology.md).

## Prace pokrewne

[FlyHack](https://arxiv.org/abs/2610.13018) (Dhawan, 2026) usuwa te same 604 klasy połączeń w tych samych
modelach flyvis 000-009 i ocenia względem tej prawdy integrated gradients i edge attribution patching;
wyczerpująca ablacja jako prawda nie jest tu nowością. Co dodaje ten benchmark: ile odzyskuje sama anatomia na
jednej skali udziału sygnału, drabinkę tanich przybliżeń (wagi, aktywność z wagami i bez, ten SAE, gradient), sieci
wytrenowane kontra niewytrenowane z obaloną hipotezą i ranking. Odczyty i prawdy są różne (FlyHack: jedna
zdekodowana liczba przepływu optycznego na zachowanie, prawda = górne 20, średnia po 20 parach model x zachowanie;
tutaj: centralna kolumna każdego typu komórki, prawda = górne 61), więc patching z zerowaniem osiąga tam AUROC
0.927, a tutaj 0.987; wyniki się uzupełniają, a nie przeczą sobie. Benchmarki ze
znaną odpowiedzią w modelach językowych: [MIB](https://arxiv.org/abs/2504.13151) i
[InterpBench](https://arxiv.org/abs/2407.14494). Więcej, także Beiran i Litwin-Kumar o tym, dlaczego konektom
nie wyznacza dynamiki: [docs/related_work.md](docs/related_work.md).

## Odtwarzalność

- **Reguły przed danymi.** Każdą regułę decyzji spisaliśmy przed danymi, których dotyczy, a każdą późniejszą
  zmianę wymieniamy z datą i powodem. Historia z datami jest w prywatnym repozytorium. Publikujemy sumy sha256
  oryginalnych dokumentów: pozwalają każdemu sprawdzić dokumenty, gdy historia z datami zostanie udostępniona;
  zewnętrzny znacznik czasu (OSF) jest planowany. Zob. [docs/preregistration.md](docs/preregistration.md).
- **Odtworzenie bajt w bajt.** `fcb reproduce --check` przelicza `results/results.json` i `docs/results.md` i
  porównuje je bajt w bajt; CI sprawdza tę ścieżkę kodu na danych syntetycznych przy każdym pushu.
- **Sumy kontrolne.** Asset wydania ma `SHA256SUMS`; `fcb download` sprawdza archiwum i każdy plik w środku
  (sha256 archiwum 1.1.0 `sha256:f21ee4573834dbfad4517da00da200c590df9f544a0d26df8d26b24f48db2bb3`, archiwum 1.0.0 `sha256:61d15c127b488eb96d0352d60e09f9f55349c91a2dac61dd90cfa236e53c1341`). Pliki pomiarów są
  bajtowo identyczne z tymi, z których policzono tabele badania.
- **Determinizm.** Stałe ziarna (bootstrap 20261101, losowe kierunki dla każdego bodźca), posortowana kolejność
  modeli, wykresy bez znaczników czasu. Użyliśmy dwóch urządzeń (Apple silicon dla modeli 000-004, CUDA dla
  005-009 i drugiego zestawu bodźców); zgodność między nimi sprawdziliśmy przed użyciem (zob.
  [docs/preregistration.md](docs/preregistration.md#deviations)).

## Wyniki negatywne i ograniczenia

- **Obalone: trening zmniejsza przewagę patchingu nad anatomią, a nie ją zwiększa.** Miara to Spearman wewnątrz
  górnych 61, jedyna, która nie nasyca się w sieciach niewytrenowanych: przewaga w sieciach wytrenowanych minus
  przewaga w niewytrenowanych wynosi **-0.705** [-0.752, -0.660 | t -0.779, -0.630]. Na AUROC przewaga wygląda
  po treningu na większą o 0.091 tylko dlatego, że patching dochodzi do sufitu w sieciach niewytrenowanych.
- **Jedyną prawdą jest zerowanie.** Patching to linearyzacja wyzerowania klasy, więc część jego 0.987 może brać
  się z dopasowania do interwencji, a nie z trafiania w to, co ważne; to benchmark metod względem ablacji, nie
  dowód „funkcji poza schematem połączeń”. Z ablacją do średniej zamiast zerowania kolejność metod się utrzymuje ([v1.1](docs/v1.1/mean-ablation.md)); inne
  interwencje nie były sprawdzane.
- **Efekt zmienia się między modelami.** Mediana Spearmana efektu między dwoma modelami to 0.33, a tylko 3 klasy są w
  górnych 61 co najmniej 9 modeli. Rankingi metod trzymają się na średniej z zespołu; efekt jednej klasy w jednym
  modelu nie jest stabilną wielkością ([Z3](docs/v1.1/z3.md)).
- **Wyciszenie całego typu może rozbić dynamikę.** W 12 z 630 par typ x model efekt jest ponad 1000 razy większy od
  mediany ([Z6](docs/v1.1/z6.md)); wnioski opieramy na rangach.
- **Prawda zadaniowa jest mała.** 16 klipów Sintel, jeden dekoder na model ([Z4](docs/v1.1/z4.md)).
- **Tanie przybliżenia nie są za darmo.** Gradient (jedno przejście wstecz) traci do patchingu 0.026 AUROC
  [-0.033, -0.019], 10 modeli. Ten SAE (0.873) nie bije zwykłej sumy |wag| (0.878); to jedna konfiguracja, nie
  rzadkie autoenkodery w ogóle.
- **„Niewytrenowana” nie znaczy „losowa”.** Inicjalizacja flyvis ustawia sumę wag każdej klasy na 0.01 x liczba
  jej krawędzi, więc sieci niewytrenowane już niosą strukturę liczby synaps (Spearman z liczbą synaps 0.662).
- **Sieć z przetasowanymi wagami jest nieustalona.** Każdy prerejestrowany sposób permutowania wytrenowanych wag
  rozregulowywał dynamikę w 4 lub 5 z 5 modeli, więc nie ma dla niej prawdy z ablacji.
- **Jedna rodzina modeli.** Jeden zespół, 10 modeli, jeden odczyt (kolumny centralne), 40 + 24 bodźce; drugi
  zestaw bodźców nie ma naturalnych filmów. Wnioski dotyczą tego modelu, nie muszki. Pełna lista:
  [docs/limitations.md](docs/limitations.md).

## Plan

- **Druga rodzina modeli.** Jeden wytrenowany zespół to najsłabszy punkt v1.0.0. Następny krok: model z
  neuronami typu leaky integrate-and-fire zbudowany z całego konektomu FlyWire (Shiu i in., Nature 2024), z tą
  samą prawdą z ablacji i tym samym rankingiem.
- **Ablacja do średniej w paczce:** pomiary ablacji do średniej i ich krok w `fcb measure` (dziś wynik z v1.1
  pochodzi z repozytorium badawczego i nie jest tu liczony od nowa).
- **Zewnętrzny znacznik czasu** dla reguł następnego badania (OSF), zanim powstaną dane, i DOI z Zenodo dla
  wydania.
- **Kod pomiaru drugiego zestawu bodźców** (przesuwane paski). v1.0.0 zawiera kod pomiaru prawdy i każdej metody
  z rankingu (`fcb measure --what all`); drugi zestaw bodźców jest tylko w paczce.
- **Twoja metoda.** Pull requesty z nowymi wierszami są mile widziane, zob. [Dodaj swoją metodę](#dodaj-swoją-metodę).

## Układ repozytorium

```
src/flyvis_circuit_bench/   metryki, statystyka, analiza, ocena metod, raport, wykresy, CLI, pobieranie
src/.../measure/            opcjonalna ścieżka pomiaru (flyvis + torch)
results/results.json        wszystkie liczby w formie maszynowej
notebooks/                  ocena własnej metody (działa w Colab)
docs/                       metodologia, prerejestracja, wyniki, ograniczenia, prace pokrewne, dane, wykresy
tests/                      testy jednostkowe, testy end-to-end na danych syntetycznych, kontrola bajtowa na danych
```

## Cytowanie

Jeśli korzystasz z benchmarku, pomiarów albo kodu, prosimy o cytowanie (zob. też [CITATION.cff](CITATION.cff)):

```bibtex
@software{szczepanik_2026_flyvis_circuit_bench,
  author  = {Szczepanik, Natalia},
  title   = {flyvis-circuit-bench: a benchmark of circuit-discovery methods
             against ablation ground truth in the flyvis connectome model},
  year    = {2026},
  version = {1.0.1},
  url     = {https://github.com/natiixnt/flyvis-circuit-bench}
}
```

DOI (Zenodo) dodamy w następnym wydaniu. Prosimy też o cytowanie flyvis: Lappalainen, J. K. i in.
Connectome-constrained networks predict neural activity across the fly visual system. *Nature* 634, 1132-1140
(2024). https://doi.org/10.1038/s41586-024-07939-3

## Licencja

Kod: [MIT](LICENSE). Pomiary, tabele wyników, wykresy i dokumentacja: [CC BY 4.0](LICENSE-data). Wkład mile
widziany, zob. [CONTRIBUTING.md](CONTRIBUTING.md) i [Kodeks postępowania](CODE_OF_CONDUCT.md).
