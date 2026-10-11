package io.github.ryders0502.yunaviewer

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class OutfitTest {
    private var next = 0L

    private fun entry(top: String = "none", bottom: String = "none", onepiece: String = "none", name: String = "x.jpg", visible: String = "upperbody") =
        OutfitEntry(
            Photo(++next, name, next, 1, 1, 1),
            JSONObject(mapOf("visible" to visible, "top" to top, "bottom" to bottom, "onepiece" to onepiece, "outer" to "none")),
            filenameOutfit(name),
        )

    @Test
    fun sameOutfitRules() {
        val shirt = entry("light blue button-up shirt", "black pleated mini skirt")
        // wording differs, colour family and garment family agree; a hidden skirt is no reason to reject
        assertTrue(sameOutfit(shirt, entry("pale blue button-up blouse", "not visible")))
        // another colour is another garment
        assertFalse(sameOutfit(shirt, entry("white button-up shirt", "black pleated mini skirt")))
        // a dress never equals top + skirt
        assertFalse(sameOutfit(shirt, entry(onepiece = "light blue knit dress")))
        // same file-name label wins over descriptions
        val a = entry("red knit sweater", name = "v5_redknit_upperbody_4x5_1k_01.jpg")
        val b = entry("white tank top", name = "v5_redknit_fullbody_4x7_1k_02.jpg")
        assertTrue(sameOutfit(a, b))
    }

    @Test
    fun clustersAndLabels() {
        val full = entry("light blue button-up shirt", "black pleated mini skirt", visible = "fullbody")
        val clusters = clusterOutfits(listOf(entry("light blue shirt", "not visible"), entry(onepiece = "red knit dress"), full))
        assertEquals(2, clusters.size)
        // the full-body shot represents its outfit
        assertEquals(full, clusters[0][0])
        assertEquals(2, clusters[0].size)
        assertEquals("연파랑 셔츠 · 검정 미니 스커트", outfitLabel(full))
        assertEquals("빨강 니트 원피스", outfitLabel(clusters[1][0]))
        assertEquals("redknit", filenameOutfit("v5_redknit_fullbody_4x7_1k_02.jpg"))
    }

    @Test
    fun manualGroupsWin() {
        val a = entry("light blue shirt", "black skirt").copy(manual = "출근룩")
        val b = entry("red knit dress").copy(manual = "출근룩")
        val c = entry("light blue shirt", "black skirt")
        // hand-made: same name groups unlike clothes, and keeps out a look-alike that was not picked
        assertTrue(sameOutfit(a, b))
        assertFalse(sameOutfit(a, c))
        assertEquals("출근룩", outfitLabel(a))
        assertEquals(2, clusterOutfits(listOf(a, b, c)).size)
    }

    @Test
    fun outfitWordsInKorean() {
        assertEquals("검정 흰색 줄무늬 상의 반바지", outfitWords("blackwhitestripedtopshorts"))
        assertEquals("곰 니트 스웨터", outfitWords("bearknitsweater"))
        assertEquals("파랑 체크 미니원피스", outfitWords("bluecheckeredminidress"))
        // unknown letters stay as one word
        assertEquals("핑크 zzq 비키니", outfitWords("pinkzzqbikini"))
        // a file-name label wins over the AI description in the title
        assertEquals("골드 드레스", outfitLabel(entry("white tank top", name = "v5_goldgown_fullbody_4x7_1k_01.jpg")))
    }
}
