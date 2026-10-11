package io.github.ryders0502.yunaviewer

import org.json.JSONObject
import kotlin.math.min

// Port of the PC library.py outfit grouping: works on stored AI descriptions, no AI calls.

private val EMPTY = setOf("none", "not visible", "")
private val STOP = setOf("a", "an", "and", "with", "the", "of", "in", "on", "worn", "over", "style")
private val WORD = Regex("[a-z0-9]+(?:-[a-z0-9]+)*")

// colour families: cream/ivory/off-white count as white, tan as beige, and so on
private val COLOR_FAMILIES = mapOf(
    "white" to "white", "cream" to "white", "ivory" to "white", "off-white" to "white", "offwhite" to "white",
    "beige" to "beige", "tan" to "beige", "camel" to "beige", "khaki" to "beige", "nude" to "beige",
    "black" to "black", "gray" to "gray", "grey" to "gray", "charcoal" to "gray", "silver" to "gray",
    "red" to "red", "burgundy" to "red", "maroon" to "red", "crimson" to "red", "pink" to "pink", "blush" to "pink",
    "orange" to "orange", "rust" to "orange", "yellow" to "yellow", "gold" to "yellow", "golden" to "yellow",
    "mustard" to "yellow", "green" to "green", "olive" to "green", "mint" to "green", "sage" to "green",
    "blue" to "blue", "navy" to "blue", "denim" to "blue", "teal" to "teal", "purple" to "purple",
    "lavender" to "purple", "violet" to "purple", "brown" to "brown",
)

// garment nouns by family; shirt/blouse/top are one family because the describer mixes them
private val GARMENT_FAMILIES = mapOf(
    "dress" to "dress", "gown" to "dress", "romper" to "romper", "jumpsuit" to "romper",
    "shirt" to "shirt", "blouse" to "shirt", "top" to "shirt", "camisole" to "shirt", "tank" to "shirt", "bustier" to "shirt",
    "jersey" to "tee", "tee" to "tee", "t-shirt" to "tee", "tshirt" to "tee",
    "sweater" to "knit", "pullover" to "knit", "cardigan" to "cardigan", "hoodie" to "hoodie", "jacket" to "jacket",
    "coat" to "jacket", "blazer" to "jacket", "skirt" to "skirt", "shorts" to "shorts", "jeans" to "pants",
    "pants" to "pants", "trousers" to "pants", "leggings" to "pants", "bikini" to "swim", "lingerie" to "swim",
    "bra" to "swim", "bralette" to "swim",
)

private val STEMS = mapOf("knitted" to "knit", "ribbed" to "rib", "ruffled" to "ruffle", "pleated" to "pleat",
    "printed" to "print", "striped" to "stripe", "embroidered" to "embroider", "cropped" to "crop")

private fun blank(text: String) = text.trim().lowercase().let { it in EMPTY || it.startsWith("not visible") }

private fun tokens(text: String): Set<String> =
    if (blank(text)) emptySet() else WORD.findAll(text.trim().lowercase()).map { it.value }.toSet() - STOP

private fun families(tokens: Set<String>, table: Map<String, String>) = tokens.mapNotNull { table[it] }.toSet()

private fun normalize(tokens: Set<String>) = tokens.map {
    COLOR_FAMILIES[it]?.let { c -> "c:$c" } ?: GARMENT_FAMILIES[it]?.let { g -> "g:$g" } ?: STEMS[it] ?: it
}.toSet()

private fun similarGarment(a: Set<String>, b: Set<String>): Boolean {
    if (a.isEmpty() || b.isEmpty()) return false
    val colorsA = families(a, COLOR_FAMILIES)
    val colorsB = families(b, COLOR_FAMILIES)
    if (colorsA.isNotEmpty() && colorsB.isNotEmpty() && (colorsA intersect colorsB).isEmpty()) return false
    val nounsA = families(a, GARMENT_FAMILIES)
    val nounsB = families(b, GARMENT_FAMILIES)
    if (nounsA.isNotEmpty() && nounsB.isNotEmpty() && (nounsA intersect nounsB).isEmpty()) return false
    val na = normalize(a)
    val nb = normalize(b)
    val common = (na intersect nb).size.toFloat()
    return common / (na union nb).size >= 0.5f || common / min(na.size, nb.size) >= 0.6f
}

private fun JSONObject.text(key: String) = optString(key)

private fun parts(tags: JSONObject): Triple<Set<String>, Set<String>, Set<String>> {
    val main = tokens(tags.text("onepiece")).ifEmpty { tokens(tags.text("top")) }
    return Triple(main, tokens(tags.text("bottom")), tokens(tags.text("outer")))
}

/** myshare names organized photos <prefix>_<outfit>_<shot>_<W>x<H>_<n>k_<seq>.<ext>; the outfit token, if any. */
private val OUTFIT_NAME = Regex("_([a-z0-9]+)_(?:headshot|upperbody|fullbody)_[0-9]+x[0-9]+_[1-4]k_[0-9]+\\.[^.]+$", RegexOption.IGNORE_CASE)

fun filenameOutfit(name: String) = OUTFIT_NAME.find(name)?.groupValues?.get(1)?.lowercase()

/** Compare only what is visible in both photos; a dress never equals top + skirt. Same file-name label = same outfit. */
fun sameOutfit(a: OutfitEntry, b: OutfitEntry): Boolean {
    // a hand-made group is exactly the photos the user put in it
    if (a.manual != null || b.manual != null) return a.manual == b.manual
    if (a.label != null && a.label == b.label) return true
    val ta = a.tags ?: return false
    val tb = b.tags ?: return false
    val (mainA, bottomA, outerA) = parts(ta)
    val (mainB, bottomB, outerB) = parts(tb)
    if (!similarGarment(mainA, mainB)) return false
    if (bottomA.isNotEmpty() && bottomB.isNotEmpty() && !similarGarment(bottomA, bottomB)) return false
    if (outerA.isNotEmpty() && outerB.isNotEmpty() && !similarGarment(outerA, outerB)) return false
    return true
}

/** A photo with its AI description, file-name outfit label and/or hand-made group name; photos with none are not grouped. */
data class OutfitEntry(val photo: Photo, val tags: JSONObject?, val label: String?, val manual: String? = null) {
    val groupable get() = tags != null || label != null || manual != null
}

/** Greedy clustering; richer descriptions (full body first) become cluster representatives. */
fun clusterOutfits(entries: List<OutfitEntry>): List<List<OutfitEntry>> {
    fun richness(e: OutfitEntry): Int {
        val t = e.tags ?: return 0
        val (main, bottom, outer) = parts(t)
        return (if (t.text("visible") == "fullbody") 10_000 else 0) + main.size + bottom.size + outer.size
    }
    val clusters = mutableListOf<MutableList<OutfitEntry>>()
    for (entry in entries.sortedByDescending(::richness)) {
        clusters.firstOrNull { sameOutfit(it[0], entry) }?.add(entry) ?: clusters.add(mutableListOf(entry))
    }
    return clusters
}

private val GARMENT_COLORS = listOf(
    "dark brown" to "진갈색", "light blue" to "연파랑", "light pink" to "연분홍",
    "off-white" to "아이보리", "ivory" to "아이보리", "cream" to "크림색",
    "navy" to "네이비", "burgundy" to "버건디", "mustard" to "머스타드",
    "black" to "검정", "white" to "흰색", "gray" to "회색", "grey" to "회색",
    "brown" to "갈색", "red" to "빨강", "blue" to "파랑", "green" to "초록",
    "pink" to "분홍", "yellow" to "노랑", "purple" to "보라", "orange" to "주황",
    "beige" to "베이지", "gold" to "금색", "silver" to "은색",
)
private val GARMENT_PATTERNS = listOf(
    "cherry blossom" to "벚꽃무늬", "floral" to "꽃무늬", "flower" to "꽃무늬",
    "striped" to "스트라이프", "stripe" to "스트라이프", "plaid" to "체크",
    "checkered" to "체크", "polka-dot" to "도트", "polka dot" to "도트",
    "patterned" to "패턴", "print" to "프린트",
)
private val GARMENT_DETAILS = listOf(
    "off-the-shoulder" to "오프숄더", "off shoulder" to "오프숄더",
    "long-sleeve" to "긴팔", "long sleeve" to "긴팔", "short-sleeve" to "반팔",
    "short sleeve" to "반팔", "sleeveless" to "민소매", "strapless" to "스트랩리스",
    "halter" to "홀터넥", "cropped" to "크롭", "crop" to "크롭", "mini" to "미니",
    "long" to "롱", "flared" to "플레어", "ruffle" to "프릴", "high slit" to "하이슬릿",
    "cutout" to "컷아웃", "sheer" to "시스루", "traditional" to "전통",
    "lace" to "레이스", "leather" to "가죽", "denim" to "데님", "satin" to "새틴",
    "knit" to "니트", "ribbed" to "골지",
)
private val GARMENT_TYPES = listOf(
    "kimono dress" to "기모노 원피스", "yukata dress" to "유카타 원피스",
    "mini dress" to "미니 원피스", "sundress" to "선드레스", "one-piece" to "원피스",
    "bodysuit" to "보디수트", "jumpsuit" to "점프수트", "romper" to "롬퍼",
    "tank top" to "탱크톱", "t-shirt" to "티셔츠", "tee shirt" to "티셔츠",
    "kimono" to "기모노", "yukata" to "유카타", "dress" to "원피스",
    "cardigan" to "가디건", "jacket" to "재킷", "blazer" to "블레이저", "coat" to "코트",
    "sweater" to "스웨터", "hoodie" to "후드", "blouse" to "블라우스", "shirt" to "셔츠",
    "camisole" to "캐미솔", "bra" to "브라", "top" to "상의",
    "mini skirt" to "미니 스커트", "skirt" to "스커트", "denim shorts" to "데님 반바지",
    "shorts" to "반바지", "jeans" to "청바지", "trousers" to "바지", "pants" to "바지",
    "thong" to "티팬티", "briefs" to "브리프", "underwear" to "속옷", "lingerie" to "란제리",
)

private fun termRegex(term: String) = Regex("(?<![a-z0-9])${Regex.escape(term)}(?![a-z0-9])")

/** Longer phrases win, labels come back in text order without duplicates. */
private fun labels(text: String, terms: List<Pair<String, String>>): List<String> {
    val candidates = terms.flatMap { (term, label) -> termRegex(term).findAll(text).map { Triple(it.range.first, it.range.last + 1, label) } }
    val chosen = mutableListOf<Triple<Int, Int, String>>()
    for (c in candidates.sortedWith(compareBy({ -(it.second - it.first) }, { it.first }))) {
        if (chosen.none { c.first < it.second && it.first < c.second }) chosen += c
    }
    return chosen.sortedBy { it.first }.map { it.third }.distinct()
}

/** "light blue cropped knit cardigan" -> "연파랑 크롭 니트 가디건" */
fun garmentLabel(text: String): String {
    val value = text.trim().lowercase()
    if (blank(value)) return ""
    val colors = labels(value, GARMENT_COLORS)
    val pattern = GARMENT_PATTERNS.firstOrNull { termRegex(it.first).containsMatchIn(value) }?.second ?: ""
    val garment = GARMENT_TYPES.firstOrNull { termRegex(it.first).containsMatchIn(value) }?.second ?: "의상"
    val details = labels(value, GARMENT_DETAILS).filter { it !in garment }
    return (colors + pattern + details + garment).distinct().filter { it.isNotEmpty() }.joinToString(" ")
}

/** The file-name outfit name in Korean when there is one (the user's own naming), else one from the AI description. */
fun outfitLabel(entry: OutfitEntry): String {
    entry.manual?.let { return it }
    entry.label?.let { return outfitWords(it) }
    val tags = entry.tags ?: return "기타 의상"
    return listOf("onepiece", "top", "bottom", "outer").map { garmentLabel(tags.text(it)) }
        .filter { it.isNotEmpty() }.joinToString(" · ").ifEmpty { "기타 의상" }
}

/** Korean for the words in myshare outfit tokens; same table as OUTFIT_WORDS in the PC index.html. */
private val OUTFIT_WORDS = mapOf(
    "white" to "흰색", "black" to "검정", "gray" to "회색", "grey" to "회색", "beige" to "베이지", "ivory" to "아이보리",
    "cream" to "크림", "red" to "빨강", "pink" to "핑크", "blue" to "파랑", "lightblue" to "하늘색", "navy" to "네이비",
    "green" to "초록", "mint" to "민트", "sage" to "세이지", "yellow" to "노랑", "mustard" to "머스터드", "orange" to "주황",
    "brown" to "갈색", "burgundy" to "버건디", "maroon" to "적갈색", "lavender" to "라벤더", "gold" to "골드", "purple" to "보라",
    "silver" to "실버", "khaki" to "카키", "denim" to "데님", "floral" to "꽃무늬", "striped" to "줄무늬", "plaid" to "체크", "checkered" to "체크",
    "gingham" to "깅엄", "polkadot" to "도트", "patterned" to "패턴", "lace" to "레이스", "satin" to "새틴", "leather" to "가죽",
    "knit" to "니트", "ribbed" to "골지", "tweed" to "트위드", "sheer" to "시스루", "ruffle" to "러플", "pleated" to "플리츠",
    "offshoulder" to "오프숄더", "oneshoulder" to "원숄더", "strapless" to "스트랩리스", "backless" to "백리스", "halter" to "홀터",
    "vneck" to "브이넥", "turtleneck" to "터틀넥", "keyhole" to "키홀", "cutout" to "컷아웃", "bodycon" to "보디콘",
    "skater" to "스케이터", "slip" to "슬립", "tie" to "타이", "trim" to "트림", "longsleeve" to "긴소매", "buttonup" to "단추",
    "collared" to "카라", "retro" to "레트로", "gothic" to "고딕", "bootcut" to "부츠컷", "bear" to "곰", "fox" to "여우",
    "maisonkitsune" to "메종키츠네", "baseball" to "야구", "shirt" to "셔츠", "tshirt" to "티셔츠", "blouse" to "블라우스",
    "top" to "상의", "croptop" to "크롭탑", "tanktop" to "나시", "tank" to "나시", "camisole" to "캐미솔", "sweater" to "스웨터",
    "cardigan" to "가디건", "vest" to "조끼", "jersey" to "저지", "bustier" to "뷔스티에", "corset" to "코르셋",
    "bodysuit" to "보디슈트", "dress" to "원피스", "minidress" to "미니원피스", "maxidress" to "맥시원피스", "sundress" to "선드레스",
    "nightdress" to "잠옷원피스", "sweaterdress" to "니트원피스", "gown" to "드레스", "romper" to "롬퍼", "skirt" to "치마",
    "mini" to "미니스커트", "miniskirt" to "미니스커트", "jeans" to "청바지", "shorts" to "반바지", "joggers" to "조거팬츠",
    "pants" to "바지", "panties" to "팬티", "bra" to "브라", "string" to "스트링", "lingerie" to "란제리", "swimsuit" to "수영복",
    "kimono" to "기모노", "set" to "세트", "loungeset" to "라운지세트", "twopiece" to "투피스", "armor" to "갑옷", "maid" to "메이드",
    "warrior" to "무사", "headshot" to "얼굴", "nude" to "누드", "bottom" to "하의", "bottoms" to "하의", "side" to "사이드",
    "ties" to "끈", "briefs" to "브리프", "underwear" to "속옷", "colored" to "컬러", "logo" to "로고", "accents" to "포인트",
    "offtheshoulder" to "오프숄더", "ruffled" to "러플", "neckline" to "넥라인", "bow" to "리본", "knitted" to "니트",
    "wet" to "젖은", "shortsleeve" to "반소매", "cowl" to "카울", "onepiece" to "원피스", "calvinklein" to "캘빈클라인",
    "jacket" to "재킷", "coat" to "코트", "trenchcoat" to "트렌치코트", "blazer" to "블레이저", "hoodie" to "후드티",
    "sweatshirt" to "맨투맨", "polo" to "폴로", "henley" to "헨리넥", "tube" to "튜브", "bikini" to "비키니", "leggings" to "레깅스",
    "slacks" to "슬랙스", "trousers" to "바지", "culottes" to "퀼로트", "overalls" to "멜빵", "jumpsuit" to "점프슈트",
    "suit" to "정장", "uniform" to "교복", "school" to "스쿨", "sailor" to "세일러", "hanbok" to "한복", "cheongsam" to "치파오",
    "qipao" to "치파오", "apron" to "앞치마", "pajama" to "잠옷", "pajamas" to "잠옷", "robe" to "가운", "scarf" to "스카프",
    "hat" to "모자", "cap" to "모자", "beret" to "베레모", "boots" to "부츠", "heels" to "힐", "socks" to "양말",
    "stockings" to "스타킹", "midi" to "미디", "maxi" to "맥시", "long" to "롱", "short" to "숏", "wide" to "와이드",
    "flared" to "플레어", "mesh" to "망사", "velvet" to "벨벳", "silk" to "실크", "chiffon" to "시폰", "linen" to "린넨",
    "cotton" to "면", "wool" to "울", "fur" to "퍼", "sequin" to "시퀸", "check" to "체크", "houndstooth" to "하운드투스",
    "camo" to "카모", "animal" to "애니멀", "leopard" to "호피", "zebra" to "지브라", "heart" to "하트", "star" to "별",
    "sleeveless" to "민소매", "puff" to "퍼프", "wrap" to "랩", "crop" to "크롭", "high" to "하이", "low" to "로우",
    "waist" to "웨이스트", "sporty" to "스포티", "sports" to "스포츠", "gym" to "운동복", "yoga" to "요가", "tennis" to "테니스",
    "golf" to "골프", "nurse" to "간호사", "police" to "경찰", "bunny" to "바니", "cat" to "고양이", "chinese" to "중국풍",
    "japanese" to "일본풍", "korean" to "한국풍", "wedding" to "웨딩", "bridal" to "브라이덜", "teal" to "청록",
    "turquoise" to "터키석", "coral" to "코랄", "peach" to "피치", "olive" to "올리브", "wine" to "와인", "charcoal" to "차콜",
    "camel" to "카멜", "tan" to "탄", "rose" to "로즈", "lilac" to "라일락", "violet" to "바이올렛", "sky" to "하늘",
    "dark" to "진한", "light" to "연한",
)

/** "blackwhitestripedtopshorts" -> "검정 흰색 줄무늬 상의 반바지": fewest unknown letters first, then fewest words (PC splitOutfit). */
fun outfitWords(token: String): String {
    class Best(val cost: Int, val words: List<String>)
    val best = arrayOfNulls<Best>(token.length + 1).also { it[0] = Best(0, emptyList()) }
    for (i in 1..token.length) {
        for (j in maxOf(0, i - 16) until i) {
            val prev = best[j] ?: continue
            val piece = token.substring(j, i)
            val known = OUTFIT_WORDS[piece]
            val cost = prev.cost + if (known != null) 1 else 100 * piece.length
            if (cost < (best[i]?.cost ?: Int.MAX_VALUE)) best[i] = Best(cost, prev.words + (known ?: piece))
        }
    }
    // merge consecutive unknown letters back into one word
    val ascii = Regex("[a-z0-9]+")
    val out = mutableListOf<String>()
    for (w in best[token.length]!!.words) {
        if (w.length == 1 && ascii.matches(w) && out.lastOrNull()?.let(ascii::matches) == true) out[out.lastIndex] += w else out += w
    }
    return out.joinToString(" ")
}
