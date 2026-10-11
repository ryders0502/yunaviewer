package io.github.ryders0502.yunaviewer

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class EditTest {
    private fun argb(r: Int, g: Int, b: Int) = (0xFF shl 24) or (r shl 16) or (g shl 8) or b

    private fun adjusted(p: EditParams, vararg px: Int) = px.copyOf().also { adjustPixels(it, px.size, 1, p) }

    @Test
    fun colors() {
        val px = intArrayOf(argb(100, 50, 200), argb(0, 0, 0))
        assertArrayEquals(px, adjusted(EditParams(), *px))
        assertEquals(argb(200, 100, 255), adjusted(EditParams(brightness = 2f), *px)[0])
        // warmer: more red, less blue, green untouched
        assertEquals(argb(125, 50, 150), adjusted(EditParams(temperature = 100f), *px)[0])
        // gray has no saturation to change
        assertEquals(argb(90, 90, 90), adjusted(EditParams(saturation = 0f), argb(90, 90, 90))[0])
        // alpha survives
        assertEquals(0x80, adjusted(EditParams(brightness = 0.5f), 0x80FFFFFF.toInt())[0] ushr 24)
    }

    @Test
    fun sharpnessLeavesFlatAreasAlone() {
        val flat = IntArray(9) { argb(120, 60, 30) }
        val out = flat.copyOf().also { adjustPixels(it, 3, 3, EditParams(sharpness = 3f)) }
        assertArrayEquals(flat, out)
    }

    @Test
    fun editNames() {
        assertEquals("a_edit.jpg", editName("a.jpg", "jpg", emptySet()))
        assertEquals("a_edit2.jpg", editName("a.jpg", "jpg", setOf("A_EDIT.jpg")))
        // editing an edited copy saves the next version of the original
        assertEquals("a_edit3.png", editName("a_edit2.png", "png", setOf("a_edit.png", "a_edit2.png")))
    }

    @Test
    fun bandLinesRoundTrip() {
        val p = EditParams(stretch = 20f, bandTop = 0.4f, bandBottom = 0.8f)
        val (top, bottom) = bandLines(p)
        assertEquals(0.4f, moveBandLine(p, top = true, at = top).bandTop, 1e-5f)
        assertEquals(0.8f, moveBandLine(p, top = false, at = bottom).bandBottom, 1e-5f)
        // no resize: lines sit where they were put
        assertEquals(0.4f to 0.8f, bandLines(p.copy(stretch = 0f)))
    }

    @Test
    fun commands() {
        val p = EditParams(brightness = 1.2f, cropL = 0.2f)
        // values are clamped to the slider range, untouched controls stay
        val bright = applyCommand(p, EditCommand(values = mapOf("contrast" to 9f), stretch = 99f), 100, 200)
        assertEquals(1.5f, bright.contrast)
        assertEquals(1.2f, bright.brightness)
        assertEquals(MAX_STRETCH_PCT, bright.stretch)
        assertEquals(EditParams(rot = 270), applyCommand(p, EditCommand(reset = true, rotate = "ccw"), 100, 200))
        // 100x200 portrait to 1:1 keeps the full width and centers the height
        val square = applyCommand(EditParams(), EditCommand(cropRatio = "1:1"), 100, 200)
        assertEquals(listOf(0f, 0.25f, 1f, 0.75f), listOf(square.cropL, square.cropT, square.cropR, square.cropB))
        // after a quarter turn the same photo is landscape
        val turned = applyCommand(EditParams(), EditCommand(rotate = "cw", cropRatio = "1:1"), 100, 200)
        assertEquals(listOf(0.25f, 0f, 0.75f, 1f), listOf(turned.cropL, turned.cropT, turned.cropR, turned.cropB))
    }

    @Test
    fun stats() {
        val s = photoStats(intArrayOf(argb(200, 100, 100), argb(0, 0, 0)))
        assertEquals(0, s["p2"])
        assertEquals(129, s["p98"])
        assertEquals(64, s["mean_saturation"])
        assertEquals(2.0, s["red_blue_ratio"])
    }

    @Test
    fun cropStaysInside() {
        val moved = moveCrop(EditParams(cropL = 0.1f, cropR = 0.5f), 4, dx = 0.9f, dy = -1f)
        assertEquals(0.6f, moved.cropL, 1e-5f)
        assertEquals(1f, moved.cropR, 1e-5f)
        assertEquals(0f, moved.cropT, 1e-5f)
        val corner = moveCrop(EditParams(), 0, dx = 2f, dy = 0f)
        assertEquals(0.95f, corner.cropL, 1e-5f)
    }
}
