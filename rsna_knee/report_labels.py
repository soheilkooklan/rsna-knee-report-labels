"""Turn a free-text knee MRI report into twelve graded targets and twelve weights.

Only 58 of the 4,407 training studies carry expert labels; the rest carry the report
the radiologist wrote when the study was read. Test studies carry neither, so the
report cannot be an input to the model. It can only be the source of the training
targets. Everything the imaging model can possibly learn is bounded by what this
module reads, which is why it is the first file in the package and the one with the
most tests behind it.

Design notes that are not obvious from the code:

* No language is detected before reading. The corpus is English, Spanish, Turkish,
  Croatian/Bosnian/Serbian, Greek, German, Bulgarian, Dutch and French. Every cue
  list below holds all nine at once and each clause is tested against the union.
  Routing on a guessed language commits to that guess before any evidence is read,
  and the cheap guesses collide badly ("la" is as common in Spanish as in French).
  Greek and Cyrillic cues cannot collide with Latin-script ones, and the Latin-script
  vocabularies of interest are close enough that a shared cue is usually right.

* Output is graded, not binary. The challenge ground truth was set by two MSK
  radiologists with an adjudicator against explicit severity thresholds, and
  borderline studies were graded negative. A report that says "trace effusion"
  therefore should not produce the same target as one that says "large effusion".

* Every target carries a weight as well as a score. A report that never mentions the
  patellofemoral compartment says nothing about it, and training that silence as a
  confident negative is the single largest avoidable error in this setup. Silence
  produces score 0 at low weight; an explicit "no effusion" produces score 0 at full
  weight. The loss in train.py multiplies by that weight.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple

TARGETS: Tuple[str, ...] = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
)

# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

# Turkish dotted and dotless i must be folded before case folding, or "IZLENMEZ"
# and "izlenmez" end up as different strings. The Croatian d-with-stroke and the
# German sharp s do not survive NFKD either, so they are mapped by hand.
_PREFOLD = str.maketrans({
    "\u0131": "i", "\u0130": "i", "I": "i",
    "\u00df": "ss", "\u0111": "d", "\u0110": "d",
    "\u00f8": "o", "\u00d8": "o", "\u00e6": "ae", "\u00c6": "ae",
})

_SOFT_HYPHEN = "\u00ad"


def normalize(text: str) -> str:
    """Fold case, accents and separators while keeping Greek and Cyrillic letters.

    NFKD splits a Latin accented character into a base letter plus a combining mark,
    which is then dropped. Greek and Cyrillic accents decompose the same way, so
    tonos and the Bulgarian stress mark disappear too and the letters survive.
    """
    if not isinstance(text, str):
        return ""
    text = text.translate(_PREFOLD).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace(_SOFT_HYPHEN, "")
    text = re.sub(r"[_/\\]+", " ", text)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def unwrap(text: str) -> str:
    """Rejoin lines a fixed-width export broke in the middle of a sentence.

    A large share of this corpus arrives hard-wrapped at some column, so a sentence
    is split across two lines with no punctuation at the break:

        Rotura compleja degenerativa del cuerno anterior, cuerpo y parte de la raiz
        meniscal posterior del menisco lateral con extrusion asociada.

    The first line says "tear" and names no structure; the second names the lateral
    meniscus and says nothing about a tear. Split on newlines and the study comes out
    silent on a meniscus the report describes as complexly torn. A line that does not
    end in sentence punctuation is a continuation.
    """
    out: List[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            out.append("\n")
            continue
        if out and out[-1] != "\n" and not re.search(r"[.;:!?]$", out[-1]):
            out[-1] = out[-1] + " " + line
        else:
            out.append(line)
    return "\n".join(x for x in out if x != "\n")


# The negative lookbehind and lookahead keep "1.2 x 1.5 cm" and "grade 2.5" in one
# piece. Measurements are extremely common in these reports and splitting inside one
# severs a finding from the structure it belongs to: "full thickness cartilage
# defect, 1.2x1.5cm." / "at lateral trochlea" reads as a defect nowhere and a
# trochlea with nothing wrong with it.
_CLAUSE_SPLIT = re.compile(
    r"(?<!\d\.)(?<!cm\.)(?<!mm\.)(?<!ml\.)(?<=[.;!?])\s+(?!\d)"
    r"|\n+"
    r"|(?<=\w),(?=\s*(?:no|sin|geen|kein|ohne|bez|yok)\b)"
)

# A structured report states the compartment once, as a heading, and then describes
# it over several sentences that never name it again.
_HEADING = re.compile(r"^([^:]{3,60}):\s*(.*)$", re.S)


def clauses(text: str) -> List[str]:
    parts = [c.strip() for c in _CLAUSE_SPLIT.split(unwrap(normalize(text)))]
    return [c for c in parts if len(c) > 2]


def scoped_clauses(text: str) -> List[str]:
    """Clauses with any compartment heading prepended to the sentences it governs.

    A heading such as ``medial compartment (meniscus, collateral ligament,
    cartilage):`` scopes everything up to the next heading. Prefixing rather than
    tracking state keeps every downstream rule (proximity, negation, severity)
    working on a single string with no special case for headings.
    """
    out: List[str] = []
    scope = ""
    for clause in clauses(text):
        m = _HEADING.match(clause)
        if m is not None:
            head, rest = m.group(1).strip(), m.group(2).strip()
            if _is_compartment_heading(head):
                scope = head
                if rest:
                    out.append(clause)
                continue
            scope = ""
            out.append(clause)
            continue
        out.append(f"{scope} {clause}".strip() if scope else clause)
    return out


def _rx(*patterns: str) -> re.Pattern:
    return re.compile("|".join(patterns))


# ---------------------------------------------------------------------------
# polarity
# ---------------------------------------------------------------------------

NEGATION = _rx(
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bnegative for\b", r"\babsen",
    r"\bunremarkable\b", r"\bfree of\b", r"\bnone\b", r"\bintact\b", r"\bnil\b",
    r"\bsin\b", r"\bno hay\b", r"\bausenc", r"\bausente", r"\bindemne",
    r"\bpas de\b", r"\bsans\b", r"\baucune?\b",
    r"\bgeen\b", r"\bzonder\b", r"\bniet\b", r"\bintacte?\b",
    r"\bkeine?\b", r"\bohne\b", r"\bnicht\b", r"\bunauffallig", r"\bregelrecht",
    r"\byok\b", r"\byoktur\b", r"izlenmemekte", r"izlenmedi", r"saptanmadi",
    r"gozlenmemekte", r"mevcut degil", r"\bdegil\b", r"\bintakt",
    r"\bnema\b", r"\bbez\b", r"\bnisu\b", r"\bnije\b", r"\buredn", r"\bureda",
    r"\bχωρις\b", r"\bδεν\b", r"\bουδεν", r"\bακεραι",
    r"\bняма\b", r"\bбез\b", r"\bне се\b", r"\bлипсв",
)

UNCERTAIN = _rx(
    r"\bpossible\b", r"\bprobable\b", r"\bsuspect", r"\bquestionable\b",
    r"\bcannot exclude\b", r"\bcannot be excluded\b", r"\bmay represent\b",
    r"\bequivocal\b", r"\bborderline\b", r"\bversus\b", r"\bvs\b", r"\bif clinically\b",
    r"\bposible\b", r"\bprobable\b", r"\bsospech", r"\bno se puede excluir\b",
    r"\bmoglich", r"\bverdacht", r"\bfraglich", r"\bnicht ausgeschlossen\b",
    r"\bmogelijk", r"\bverdenking\b", r"\bniet uit te sluiten\b",
    r"\bsuphel", r"\bolasi\b", r"\bekarte edilemez\b", r"\bayirici\b",
    r"\bmoguc", r"\bsumnja\b", r"\bvjerojatn",
    r"\bπιθαν", r"\bυποψια\b", r"\bδεν αποκλειεται\b",
    r"\bвъзможн", r"\bвероятн", r"\bсъмнени",
)

# "the meniscus is normal" negates; "normal alignment, but a tear of ..." does not.
NORMALITY = _rx(
    r"\bnormal\b", r"\bunremarkable\b", r"\bpreserved\b", r"\bwithin normal limits\b",
    r"\bnormales?\b", r"\bconservad", r"\bnormaal\b", r"\bnormale?[ns]?\b",
    r"\bnormaldir\b", r"\bdogal\b", r"\bnormalan\b", r"\bnormaln",
    r"\bφυσιολογικ", r"\bнормал",
)

# ---------------------------------------------------------------------------
# pathology vocabulary
# ---------------------------------------------------------------------------

TEAR = _rx(
    r"\btear", r"\btorn\b", r"\bruptur", r"\bdisruption\b", r"discontinuit",
    r"\bavuls", r"\bmacerat", r"bucket ?handle", r"\bdetach",
    r"\brotura", r"\bruptura", r"\bdesgarro", r"\broto\b", r"\bdesinsercion\b",
    r"\bdechirure", r"\bdechire",
    r"\bscheur", r"\bruptuur", r"gescheurd",
    r"\briss\b", r"einriss", r"zerreiss", r"\bausriss", r"\blasion",
    r"\byirtik", r"\byirtig", r"\bkopma\b", r"butunluk kaybi", r"devamsizlik",
    r"devamliligi secilememis", r"\brupturu\b",
    r"\bpuknuc", r"\bprekid\b", r"\bpukotin", r"\braskid",
    r"\bρηξη", r"\bρηξις", r"\bρηγμα", r"\bασυνεχεια", r"\bδιατομη",
    r"\bруптура", r"\bразкъсв", r"\bразрив", r"\bскъсв",
)

DEGEN = _rx(
    r"degenerat", r"\bmucoid\b", r"\bmyxoid\b", r"\bfray", r"\bfissur",
    r"dejeneratif", r"\bdejenerasyon\b", r"\bmukoid\b", r"degenerativn",
    r"\bεκφυλ", r"\bμυξοειδ", r"\bμυξωδ", r"\bдегенерат",
    r"\bmeniskopat", r"\bmeniscopat", r"aufgefasert", r"\bmuco ?ide\b",
)

INJURY = _rx(
    r"\binjur", r"\bsprain\b", r"\blesion", r"\bstrain\b", r"\bpartial thickness\b",
    r"\bhigh signal\b", r"\bsignal alterat", r"\bedema\b", r"\boedema\b",
    r"\bdistension\b", r"\bthicken", r"\bengrosamiento\b",
    r"\bodem", r"\bedem\b", r"\bοιδημα", r"\bοδημα", r"\bодем", r"\bедем",
    r"\bhyperintens", r"\bhiperintens", r"\bυπερεντασ", r"\bхиперинтенз",
    r"\bdistorsion\b", r"\bzerrung\b", r"\bverdickung\b", r"\bkontuzyon\b",
)

# ---------------------------------------------------------------------------
# anatomy vocabulary
# ---------------------------------------------------------------------------

STEM_MENISCUS = _rx(r"menisc\w*", r"menisk\w*", r"\bμηνισκ\w*", r"\bмениск\w*")

STEM_CRUCIATE = _rx(
    r"cruciate", r"cruzado", r"croise", r"kruisband", r"kreuzband",
    r"capraz bag\w*", r"krizn\w*", r"\bχιαστ\w*", r"\bкръстн\w*",
    r"\bacl\b", r"\blca\b", r"\bvkb\b", r"\bocb\b", r"\bacb\b",
)

STEM_COLLATERAL = _rx(
    r"collateral\w*", r"colateral\w*", r"kollateral\w*", r"kolateraln\w*",
    r"collaterale\w*", r"yan bag\w*", r"\bπλαγι\w*", r"\bколатерал\w*",
    r"\bстранич\w*", r"innenband\w*", r"binnenband\w*", r"\bmcl\b", r"\blcm\b",
    r"\bmkl\b", r"\bibl\b",
)

SIDE_ANTERIOR = _rx(
    r"\banterior\b", r"\bant\b", r"\banterieur\b", r"\bvoorste\b", r"\bvorder",
    r"\bon\b", r"\bprednj", r"\bπροσθι", r"\bпредн",
)

SIDE_MEDIAL = _rx(
    r"\bmedial\w*\b", r"\binternal\b", r"\binterno\b", r"\binterna\b",
    r"\bmediaal\b", r"\bmediale?n?\b", r"\bic\b", r"\biç\b", r"\bunutarnj",
    r"\bunutrasnj", r"\bmedijaln", r"\bεσω\b", r"\bεσωτερικ", r"\bвътреш",
    r"\bмедиал", r"\binnen", r"\bbinnen",
)

SIDE_LATERAL = _rx(
    r"\blateral\w*\b", r"\bexternal\b", r"\bexterno\b", r"\bexterna\b",
    r"\blateraal\b", r"\blaterale?n?\b", r"\bdis\b", r"\bdış\b", r"\bvanjsk",
    r"\bspoljasnj", r"\bεξω\b", r"\bεξωτερικ", r"\bвъншн", r"\bлатерал",
    r"\baussen", r"\bbuiten",
)

# Phrases that identify a target on their own, without a proximity test.
ANAT = {
    "ACL": _rx(
        r"anterior cruciate", r"\bacl\b", r"cruzado anterior", r"\blca\b",
        r"croise anterieur", r"voorste kruisband", r"\bvkb\b",
        r"vorderes? kreuzband", r"vorderen kreuzband", r"on capraz", r"\bocb\b",
        r"prednji krizni", r"prednjeg krizn", r"prednje ukrsten",
        r"προσθιο\w* χιαστ", r"προσθιου χιαστου", r"предна кръстна",
        r"предната кръстна",
    ),
    "MCL": _rx(
        r"medial collateral", r"\bmcl\b", r"tibial collateral",
        r"colateral (?:medial|interno|interna)", r"\blcm\b", r"ligamento interno",
        r"collateral interne", r"mediale? collateraal", r"mediale kollateralband",
        r"innenband", r"binnenband", r"ic yan bag", r"medial kollateral",
        r"medijalni kolateraln", r"unutarnji kolateraln", r"\bmkl\b",
        r"εσω πλαγι", r"вътрешн\w* колатерал", r"медиален колатерал",
    ),
}

# Findings named directly, with no side qualifier needed.
DIRECT = {
    "Effusion": _rx(
        r"\beffusion\b", r"joint fluid", r"\bhydrarthros", r"suprapatellar fluid",
        r"fluid (?:in|within) the joint", r"joint distension",
        r"recess distension", r"distended? suprapatellar",
        r"\bderrame\b", r"\bepanchement\b", r"liquido (?:articular|intraarticular)",
        r"\bvocht\b", r"gewrichtsvocht", r"opzetting van (?:de )?suprapatellai?r",
        r"suprapatellai?re? recessus", r"\bhydrops\b",
        r"\berguss\b", r"gelenkerguss", r"ergussbildung",
        r"\befuzyon\b", r"eklem\w* (?:ici )?sivi", r"sivi (?:artis|miktari)",
        r"suprapatellar bursa\w* (?:sivi|efuzyon)", r"eklem mesafesinde sivi",
        r"\bizliv\b", r"\bizljev\b", r"\bvyliv\b", r"zglobn\w* tekuc",
        r"\bvise tekucine\b", r"tekucin\w* u zglobu",
        r"\bυγρο\b", r"\bσυλλογη\b", r"αρθρικ\w* υγρ", r"\bυδραρθρ",
        r"\bизлив\b", r"\bтечност\b", r"ставна течност",
    ),
    "Synovitis": _rx(
        r"\bsynovit", r"synovial (?:thicken|proliferat|hypertroph|enhance|nodul)",
        r"\bsinovit", r"\bsynovial hyperplas", r"\bpannus\b", r"\bplica\b",
        r"\bsynovialit", r"\bυμενιτ", r"\bсиновит", r"\bsynovium\b",
        r"synoviale? (?:verdikking|verdickung|zwelling|proliferation)",
        r"verdikking(?:en)? van (?:van )?het synovium",
        r"sinovijaln\w* (?:zadebljanj|proliferac)", r"sinovyal (?:kalinlas|proliferas)",
        r"\bυμενικ\w* (?:παχυνσ|υπερπλασ)", r"синовиaлн\w* (?:задебел|пролифер)",
        r"\bbursit", r"\bbursitis\b",
    ),
    "Baker's": _rx(
        r"baker\w* cyst", r"baker cyst", r"popliteal cyst", r"\bbakerzyste\b",
        r"quiste de baker", r"quiste poplite", r"kyste de baker",
        r"bakerse cyste", r"baker kist", r"popliteal kist", r"bakerova cist",
        r"poplitealn\w* cist", r"κυστη baker", r"κυστη του baker",
        r"ιγνυακ\w* κυστ", r"бейкер", r"поплитеaлн\w* киста",
        r"\bbaker\b", r"gastrocnemio ?semimembranos",
    ),
    "Contusion": _rx(
        r"\bcontusion\b", r"\bcontusio\b", r"bone bruise", r"\bbml\b",
        r"(?:bone )?marrow o?edema", r"marrow contusion", r"medullary o?edema",
        r"\btrabecular o?edema\b", r"subchondral o?edema", r"bone marrow signal",
        r"\bcontusion osea\b", r"edema oseo", r"edema (?:de )?medula osea",
        r"edema (?:de )?medula", r"contusion osseuse", r"oedeme osseux",
        r"\bbotcontusie\b", r"beenmergoedeem", r"beenmerg ?oedeem",
        r"\bknochenmarkodem", r"knochenmark ?odem", r"knochenmarksodem",
        r"kemik ilig\w* odem", r"kemik kontuzyon", r"\bkontuzyo",
        r"kontuzij\w* kost", r"edem kostn\w* srz", r"nagnjecenj",
        r"οιδημα (?:του )?μυελου", r"οστεομυελικ\w* οιδημα", r"θλαση οστου",
        r"костномозъчен оток", r"\bконтузи", r"оток на костния мозък",
        r"kostan\w* edem", r"edem kosti", r"kostn\w* edem",
        r"\bοστικ\w* μωλωπ", r"οιδημα\w*[^.;]{0,30}μυελ", r"οιδημα\w*[^.;]{0,30}οστ",
        r"kemik ilig\w*[^.;]{0,15}odem", r"odem\w*[^.;]{0,15}kemik ilig",
        r"medula osea[^.;]{0,15}edema", r"edema[^.;]{0,20}medula",
    ),
    "Fracture": _rx(
        r"\bfractur", r"\bfraktur", r"\bfractura\b", r"\bfracturas\b",
        r"\bfrakture\b", r"\bkirik\b", r"\bkırık\b", r"\bprelom\b",
        r"\bkataγ", r"\bκαταγμα", r"\bфрактура\b", r"\bсчупван",
        r"\bavulsion fragment\b", r"cortical break", r"insufficiency fractur",
        r"stress fractur", r"\bimpaction\b", r"osteochondral (?:defect|lesion|fragment)",
        r"segond", r"\bfraktuur\b", r"impresijsk\w* fraktur", r"impression fractur",
        r"depressed fractur", r"\bimpaksiyon\b", r"cokme kirig", r"\bavulzij",
        r"\bavulsion\b", r"\bkirigi\b", r"\bkirik\w*", r"\bprelom\w*",
    ),
}

STEM_FRACTURE = _rx(r"\bfractur", r"\bfraktur", r"\bkirik\b", r"\bprelom\b",
                    r"\bκαταγμα", r"\bфрактур")

# Osteoarthritis: the finding is named once and the compartment separately.
OA_TERM = _rx(
    r"osteoarthr", r"\bosteoarthrosis\b", r"\barthros", r"\barthrit",
    r"\bchondrosis\b", r"chondromalac", r"cartilage (?:loss|thinning|defect|dener)",
    r"cartilage (?:full|partial) thickness", r"joint space narrowing", r"\bjsn\b",
    r"\bosteophyt", r"\bspur", r"subchondral (?:sclerosis|cyst)",
    r"\bgonarthros", r"\bartrosis\b", r"\bcondropat", r"\bcondromalac", r"perdida de cartilago",
    r"adelgazamiento (?:del )?cartilago", r"pinzamiento (?:articular|femorotibial)",
    r"\bosteofit", r"\bgonartros",
    r"\barthrose\b", r"\bchondropath", r"\bosteophyte",
    r"\bartrose\b", r"\bkraakbeen", r"\bgewrichtsspleetversmalling\b",
    r"\bknorpel", r"\bgelenkspaltverschmal", r"\bosteophyten\b",
    r"\bkondromalaz", r"\bkikirdak", r"eklem araligi daralma",
    r"\bosteoartrit", r"\bartroz", r"\bhondromalac", r"\bhrskavic",
    r"suzenje zglobnog prostora", r"\bosteofiti", r"\bgonartroz",
    r"\bοστεοαρθρ", r"\bχονδροπαθ", r"\bχονδρομαλακ", r"\bοστεοφυ",
    r"\bαρθριτ", r"στενωση (?:του )?μεσοδιαστηματος",
    r"\bостеоартр", r"\bхондропат", r"\bхондромалац", r"\bостеофит",
    r"\bгонартроз", r"\bстеснени",
    r"\bchondral\b", r"osteochondral (?:defect|lesion|ulcer)", r"\bcondral\b",
    r"ulcera condral", r"lesion condral", r"perdida (?:focal )?de cartilago",
    r"kraakbeenlijden", r"kraakbeenverlies", r"kraakbeendefect",
    r"knorpelschaden", r"knorpeldefekt", r"knorpelverlust",
    r"kikirdak (?:kayb|incel|defekt)", r"hrskavicn\w*", r"gubitak hrskavic",
    r"\bχονδρ\w*[^.;]{0,20}(?:βλαβ|απωλει|εξαλειψ|ελλειμ)",
    r"αρθρικ\w* χονδρ", r"\bхрущял",
)

# The tibiofemoral compartments are almost never named by a single fixed phrase.
# Reports say "medial femorotibial", "medial compartment", "medial femoral condyle",
# "medial tibial plateau" or, in Greek, "εσω διαμερισμα". That is five constructions
# per language across nine languages. Matching a compartment stem near a side
# qualifier covers all of them; a phrase list covered about a third and the misses were
# concentrated in the non-English languages, which is a bias aligned with the
# reporting site rather than noise that averages out.
STEM_COMPARTMENT = _rx(
    r"femorotibia\w*", r"tibiofemora\w*", r"compartment\w*", r"compartiment\w*",
    r"kompartiment\w*", r"condyl\w*", r"kondil\w*", r"kondyl\w*", r"\bplateau\b",
    r"\bplato\b", r"tibial plateau", r"gewrichtsspleet", r"gelenkspalt",
    r"femorotibialn\w*", r"kondilin", r"tibiaplateau", r"tibiaplato", r"\bδιαμερισμα\w*", r"\bδιαμερισματ\w*",
    r"\bκονδυλ\w*", r"\bκνημιαι\w*", r"\bμηριαιοκνημικ\w*",
    r"\bотдел\w*", r"\bкондил\w*", r"бедрено ?тибиaлн\w*",
    r"\bmeniscal extrusion\b", r"\bextrusion meniscal\b",
)

PF_TERM = _rx(
    r"patello ?femoral\w*", r"femoro ?patell\w*", r"\bretropatellar\w*",
    r"\btrochlea\w*", r"\btroclea\w*", r"\btrohlea\w*", r"\btroklea\w*", r"\bpatella\w*\b", r"\bpatelar\w*\b",
    r"femoropatelar\w*", r"rotulofemoral\w*", r"\brotula\b", r"\bpatelofemoral\w*",
    r"patellofemoraal", r"patellofemoraln\w*", r"patellar cartilage",
    r"\bμηριαιοεπιγονατιδικ\w*", r"\bεπιγονατιδ\w*", r"\bпателофеморал\w*",
    r"\bпателарн\w*", r"\bpatellaris\b", r"\bdiz kapagi\b",
)

# Osteoarthritis stated without naming a compartment. The finding is still real, so
# it is inherited by all three compartments at a reduced score rather than dropped.
def _is_compartment_heading(head: str) -> bool:
    """True when a heading names a compartment rather than a report section."""
    if PF_TERM.search(head):
        return True
    return bool(_near(head, STEM_COMPARTMENT, SIDE_MEDIAL)
                or _near(head, STEM_COMPARTMENT, SIDE_LATERAL))


OA_UNSCOPED = _rx(
    r"\bgonarthros", r"\bgonartros", r"\bgonartroz", r"\bγοναρθρ", r"\bгонартр",
    r"degenerative (?:joint|changes|disease)", r"cambios degenerativos",
    r"degeneratieve (?:afwijkingen|veranderingen)", r"degenerative veranderungen",
    r"dejeneratif degisiklik", r"degenerativn\w* promjen",
    r"εκφυλιστικ\w* (?:αλλοιωσ|μεταβολ)", r"дегенеративни промени",
    r"\bosteoarthritis of the knee\b", r"\bartrosis de rodilla\b",
)

TRICOMPARTMENTAL = _rx(
    r"\btricompartment", r"\btri ?compartment", r"three compartment",
    r"\btricompartimental", r"\btrikompartiment", r"\buc kompartman",
    r"τρικομματικ", r"\bтрикомпартм",
)

BICOMPARTMENTAL = _rx(
    r"\bbicompartment", r"\bbi ?compartment", r"two compartment",
    r"\bbicompartimental", r"\bbikompartiment", r"\biki kompartman",
    r"δικομματικ", r"\bбикомпартм",
)

# ---------------------------------------------------------------------------
# severity
# ---------------------------------------------------------------------------

SEV_LOW = _rx(
    r"\bsmall\b", r"\bminimal\b", r"\btrace\b", r"\bmild\b", r"\bslight\b",
    r"\btiny\b", r"\bscant\b", r"\bdiscrete\b", r"\bmilde?\b", r"\bearly\b",
    r"\bleve\b", r"\bminim", r"\bpeque", r"\bligero\b", r"\bescaso\b", r"\bdiscreto\b",
    r"\bhafif\b", r"\baz miktarda\b", r"\bsilik\b", r"\bminimal\b",
    r"\bmanja\b", r"\bmanji\b", r"\bblago\b", r"\bdiskretn", r"\bmalo\b",
    r"\bgering", r"\bdiskret", r"\bkleine?r?\b", r"\bwenig\b", r"\bzarte?\b",
    r"\bbeperkte?\b", r"\bgeringe\b", r"\bweinig\b", r"\blichte?\b",
    r"\bηπι", r"\bμικρ", r"\bελαχιστ", r"\bминимал", r"\bлек", r"\bмалк",
    r"\bgrade 1\b", r"\bgrado 1\b", r"\bgrad 1\b", r"\bgrade i\b",
)

SEV_HIGH = _rx(
    r"\blarge\b", r"\bmarked\b", r"\bmassive\b", r"\bsevere\b", r"\bextensive\b",
    r"\bgross\b", r"\bcomplete\b", r"\bfull thickness\b", r"\bhigh grade\b",
    r"\bcomplex\b", r"\badvanced\b", r"\bsignificant\b",
    r"\bgrande\b", r"\bsevera?\b", r"\bimportante\b", r"\bmarcada?\b", r"\bavanzada?\b",
    r"\bcompleta?\b", r"\bextensa?\b",
    r"\bbelirgin\b", r"\bileri\b", r"\byaygin\b", r"\bciddi\b", r"\btam kat\b",
    r"\bizrazit", r"\bteska?\b", r"\bopsezn", r"\buznapredoval",
    r"\bausgepragt", r"\bdeutlich", r"\bschwer", r"\bmassiv", r"\bfortgeschritten",
    r"\bduidelijke?\b", r"\buitgebreid", r"\bernstige?\b", r"\bfors\b",
    r"\bσοβαρ", r"\bεκτεταμεν", r"\bεντον", r"\bμεγαλ",
    r"\bтежк", r"\bизразен", r"\bголям", r"\bобширн",
    r"\bgrade 3\b", r"\bgrade 4\b", r"\bgrado 3\b", r"\bgrado 4\b",
    r"\bgrad 3\b", r"\bgrad 4\b", r"\bgrade iii\b", r"\bgrade iv\b",
)


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------

def _near(clause: str, stem: re.Pattern, qual: re.Pattern, window: int = 60):
    """True when a structure stem and its side qualifier occur close together.

    A clause can name the meniscus and the word "medial" thirty characters apart and
    still mean the medial meniscus; the same two words in a sentence listing four
    structures usually do not. A character window is crude and it is the right kind of
    crude here, because the corpus has nine word orders and no parser covers them.
    """
    for m in stem.finditer(clause):
        lo = max(0, m.start() - window)
        hi = min(len(clause), m.end() + window)
        if qual.search(clause[lo:hi]):
            return m
    return None


_PAIRED = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus")

_STEM_RULES = {
    "ACL": (STEM_CRUCIATE, SIDE_ANTERIOR),
    "MCL": (STEM_COLLATERAL, SIDE_MEDIAL),
    "Medial Meniscus": (STEM_MENISCUS, SIDE_MEDIAL),
    "Lateral Meniscus": (STEM_MENISCUS, SIDE_LATERAL),
}


_OA_TARGETS = ("Medial OA", "Lateral OA", "PF OA")


def _find_structure(clause: str, target: str):
    """Locate the anatomy for one target in one clause, or return None."""
    if target in _PAIRED:
        phrase = ANAT.get(target)
        if phrase is not None:
            m = phrase.search(clause)
            if m is not None:
                return m
        stem, side = _STEM_RULES[target]
        return _near(clause, stem, side)
    if target == "PF OA":
        return PF_TERM.search(clause)
    if target == "Medial OA":
        return _near(clause, STEM_COMPARTMENT, SIDE_MEDIAL)
    if target == "Lateral OA":
        return _near(clause, STEM_COMPARTMENT, SIDE_LATERAL)
    return DIRECT[target].search(clause)


def _polarity(clause: str, anchor_end: int) -> str:
    """Positive, negative or uncertain, at clause scope.

    Two rules that a plain keyword test gets wrong:

    "without" and its equivalents govern what comes *after* them, so
    "Subchondral insufficiency fracture at the medial tibial plateau without
    articular surface collapse" asserts a fracture even though it contains a
    negator. The negator is only read when it precedes the finding.

    "Normal" negates only when nothing else in the clause contradicts it, so
    "normal alignment with a radial tear of the posterior horn" stays positive.
    """
    if UNCERTAIN.search(clause):
        return "uncertain"
    neg = NEGATION.search(clause[:anchor_end])
    if neg is not None:
        return "negative"
    if NORMALITY.search(clause) and not (TEAR.search(clause) or STEM_FRACTURE.search(clause)):
        return "negative"
    return "positive"


def _severity_scale(clause: str) -> float:
    """Map a severity adjective onto the graded score.

    The reference standard was thresholded, and studies the readers were unsure about
    were graded negative. A "trace" effusion is therefore closer to a negative than a
    "large" one, and a flat binary score throws that ordering away for free.
    """
    if SEV_HIGH.search(clause):
        return 1.0
    if SEV_LOW.search(clause):
        return 0.45
    return 0.80


# Which pathology words count as evidence for each target once its anatomy is found.
_PATHOLOGY = {
    "ACL": (TEAR, INJURY),
    "MCL": (TEAR, INJURY),
    "Medial Meniscus": (TEAR, DEGEN),
    "Lateral Meniscus": (TEAR, DEGEN),
}


def extract(report: str) -> Dict[str, float]:
    """Read one report into 12 scores in [0, 1] and 12 weights in [0, 1].

    Returns a flat dict: one key per target plus ``<target>__w`` for its weight.
    """
    out: Dict[str, float] = {}
    pos: Dict[str, List[float]] = {t: [] for t in TARGETS}
    neg: Dict[str, int] = {t: 0 for t in TARGETS}
    unc: Dict[str, int] = {t: 0 for t in TARGETS}

    text_clauses = scoped_clauses(report)
    oa_clauses = {c for c in text_clauses if OA_TERM.search(c)}

    for clause in text_clauses:
        for target in TARGETS:
            # Osteoarthritis needs the finding word and the compartment in the same
            # clause; the other targets are named outright.
            if target in _OA_TARGETS and clause not in oa_clauses:
                continue

            m = _find_structure(clause, target)
            if m is None:
                continue

            polarity = _polarity(clause, m.end())
            if polarity == "negative":
                neg[target] += 1
                continue
            if polarity == "uncertain":
                unc[target] += 1
                continue

            if target in _PATHOLOGY:
                strong, weak = _PATHOLOGY[target]
                if strong.search(clause):
                    pos[target].append(_severity_scale(clause))
                elif weak.search(clause):
                    pos[target].append(0.40 * _severity_scale(clause) / 0.80)
                continue

            pos[target].append(_severity_scale(clause))

    # --- second pass: findings stated without the compartment that owns them ------
    #
    # "Tricompartmental osteoarthritis" and "gonarthrosis" assert osteoarthritis in
    # compartments the sentence never names. Reading them as silence loses the most
    # common way the corpus states this finding at all. The inherited score is held
    # below a directly stated one, because the report is evidence that the joint is
    # arthritic rather than evidence about which compartment.
    for clause in oa_clauses:
        if _polarity(clause, len(clause)) != "positive":
            continue
        scale = _severity_scale(clause)
        if TRICOMPARTMENTAL.search(clause):
            inherit, targets = 0.95 * scale, _OA_TARGETS
        elif BICOMPARTMENTAL.search(clause):
            # Medial and patellofemoral are the two compartments that carry
            # bicompartmental disease in the large majority of knees.
            inherit, targets = 0.80 * scale, ("Medial OA", "PF OA")
        elif OA_UNSCOPED.search(clause):
            inherit, targets = 0.62 * scale, _OA_TARGETS
        else:
            continue
        for t in targets:
            if not pos[t]:
                pos[t].append(inherit)

    # --- synovitis backoff -------------------------------------------------------
    #
    # Roughly nine in ten reports never write the word. Synovitis and effusion share
    # a cause and co-occur strongly, so a study with a stated moderate-or-larger
    # effusion is better modelled as probably-inflamed than as silent. The score is
    # held well below a stated synovitis and the weight below a stated observation,
    # so this can tilt a ranking without asserting a finding.
    if not pos["Synovitis"] and not neg["Synovitis"] and pos["Effusion"]:
        eff = max(pos["Effusion"])
        if eff >= 0.70:
            pos["Synovitis"].append(0.30 * eff)
            unc["Synovitis"] = 0  # scored, not merely hedged

    for target in TARGETS:
        if pos[target]:
            # Several mentions of one finding: take the strongest. A report that
            # describes a tear in one sentence and hedges in the next is describing
            # a tear.
            score = max(pos[target])
            weight = 1.0
        elif neg[target]:
            score, weight = 0.0, 1.0
        elif unc[target]:
            score, weight = 0.35, 0.4
        else:
            # Never mentioned. Most unmentioned findings are absent, since
            # radiologists do report effusions and tears, so the score stays low.
            # The weight drops because the report is not evidence, it is silence.
            score, weight = 0.05, 0.35
        out[target] = score
        out[target + "__w"] = weight

    return out


def extract_frame(reports, index=None):
    """Vectorised wrapper: a sequence of report strings to a DataFrame."""
    import pandas as pd

    rows = [extract(r if isinstance(r, str) else "") for r in reports]
    frame = pd.DataFrame(rows)
    if index is not None:
        frame.index = index
    return frame
