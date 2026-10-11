package io.github.ryders0502.yunaviewer

import android.content.ContentUris
import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.net.Uri
import android.provider.MediaStore.Images.Media
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

/** Same pipeline as the PC editor: orient, band stretch, crop, then color. Crop and band lines are fractions. */
data class EditParams(
    val rot: Int = 0,
    val flip: Boolean = false,
    val stretch: Float = 0f,
    val bandTop: Float = 0.5f,
    val bandBottom: Float = 0.9f,
    val cropL: Float = 0f,
    val cropT: Float = 0f,
    val cropR: Float = 1f,
    val cropB: Float = 1f,
    val brightness: Float = 1f,
    val contrast: Float = 1f,
    val saturation: Float = 1f,
    val gamma: Float = 1f,
    val temperature: Float = 0f,
    val sharpness: Float = 1f,
) {
    val cropped get() = cropL > 0f || cropT > 0f || cropR < 1f || cropB < 1f
    val adjusted get() = brightness != 1f || contrast != 1f || saturation != 1f || gamma != 1f || temperature != 0f || sharpness != 1f
}

class Control(
    val key: String,
    val label: String,
    val icon: Int,
    val range: ClosedFloatingPointRange<Float>,
    val neutral: Float,
    val get: (EditParams) -> Float,
    val set: EditParams.(Float) -> EditParams,
) {
    // multipliers read as +/-%, temperature as is
    fun text(p: EditParams) = "%+d".format((if (neutral == 1f) (get(p) - 1) * 100 else get(p)).roundToInt())
}

val CONTROLS = listOf(
    Control("brightness", "밝기", R.drawable.ic_brightness_6, 0.5f..1.5f, 1f, { it.brightness }, { copy(brightness = it) }),
    Control("contrast", "대비", R.drawable.ic_contrast, 0.5f..1.5f, 1f, { it.contrast }, { copy(contrast = it) }),
    Control("saturation", "채도", R.drawable.ic_invert_colors, 0f..2f, 1f, { it.saturation }, { copy(saturation = it) }),
    Control("gamma", "감마", R.drawable.ic_tonality, 0.5f..2f, 1f, { it.gamma }, { copy(gamma = it) }),
    Control("temperature", "색온도", R.drawable.ic_thermostat, -100f..100f, 0f, { it.temperature }, { copy(temperature = it) }),
    Control("sharpness", "선명도", R.drawable.ic_deblur, 0f..3f, 1f, { it.sharpness }, { copy(sharpness = it) }),
)

/** Color values only, geometry kept. */
fun EditParams.withColors(values: Map<String, Float>) = CONTROLS.fold(this) { p, c ->
    values[c.key]?.let { c.set(p, it.coerceIn(c.range)) } ?: p
}

fun EditParams.neutralColors() = CONTROLS.fold(this) { p, c -> c.set(p, c.neutral) }

/** What the model asked for in an editor command; null = leave as is. */
data class EditCommand(
    val values: Map<String, Float> = emptyMap(),
    val stretch: Float? = null,
    val rotate: String? = null,
    val flip: String? = null,
    val cropRatio: String? = null,
    val reset: Boolean = false,
    val autoEnhance: Boolean = false,
    val message: String = "",
)

private val RATIOS = mapOf("1:1" to 1f, "4:5" to 4 / 5f, "2:3" to 2 / 3f, "9:16" to 9 / 16f, "16:9" to 16 / 9f)

/** Applies [c] to [p]; [w] x [h] is the photo as decoded, before rotation and band stretch. */
fun applyCommand(p: EditParams, c: EditCommand, w: Int, h: Int): EditParams {
    var q = (if (c.reset) EditParams() else p).withColors(c.values)
    c.stretch?.let { q = q.copy(stretch = it.coerceIn(-MAX_STRETCH_PCT, MAX_STRETCH_PCT).roundToInt().toFloat()) }
    val full = EditParams()
    fun uncropped(x: EditParams) = x.copy(cropL = full.cropL, cropT = full.cropT, cropR = full.cropR, cropB = full.cropB)
    // the crop box is in the old orientation's coordinates, so turning starts it over
    when (c.rotate) {
        "cw" -> q = uncropped(q).copy(rot = (q.rot + 90) % 360)
        "ccw" -> q = uncropped(q).copy(rot = (q.rot + 270) % 360)
        "180" -> q = uncropped(q).copy(rot = (q.rot + 180) % 360)
    }
    when (c.flip) {
        "horizontal" -> q = q.copy(flip = !q.flip, cropL = 1 - q.cropR, cropR = 1 - q.cropL)
        "vertical" -> q = uncropped(q).copy(flip = !q.flip, rot = (q.rot + 180) % 360) // mirror + half turn
    }
    if (c.cropRatio == "original") q = uncropped(q)
    RATIOS[c.cropRatio]?.let { target ->
        val (ow, oh) = if (q.rot % 180 == 0) w to h else h to w
        val current = ow / (oh * (1 + (q.bandBottom - q.bandTop) * q.stretch / 100))
        q = if (current > target) {
            val f = target / current
            q.copy(cropL = (1 - f) / 2, cropR = (1 + f) / 2, cropT = 0f, cropB = 1f)
        } else {
            val f = current / target
            q.copy(cropL = 0f, cropR = 1f, cropT = (1 - f) / 2, cropB = (1 + f) / 2)
        }
    }
    return q
}

/** Pixel statistics the PC sends with an auto-correct request (0-255, PIL "L" luminance). */
fun photoStats(px: IntArray): Map<String, Number> {
    val hist = IntArray(256)
    var r = 0L
    var b = 0L
    var sat = 0L
    for (c in px) {
        val (cr, cg, cb) = Triple(c shr 16 and 255, c shr 8 and 255, c and 255)
        hist[luma(cr, cg, cb)]++
        r += cr
        b += cb
        val hi = maxOf(cr, cg, cb)
        if (hi > 0) sat += (hi - minOf(cr, cg, cb)) * 255 / hi
    }
    fun percentile(q: Double): Int {
        var count = 0L
        hist.forEachIndexed { level, n ->
            count += n
            if (count >= px.size * q) return level
        }
        return 255
    }
    val n = max(1, px.size)
    return mapOf(
        "mean" to (hist.withIndex().sumOf { (level, count) -> level.toLong() * count } / n.toDouble()).roundToInt(),
        "p2" to percentile(0.02), "p50" to percentile(0.5), "p98" to percentile(0.98),
        "mean_saturation" to (sat / n.toDouble()).roundToInt(),
        "red_blue_ratio" to (r / max(1.0, b.toDouble()) * 100).roundToInt() / 100.0,
    )
}

const val MAX_STRETCH_PCT = 30f // body_edit.MAX_EDIT_PCT on the PC
private const val MIN_BAND = 0.05f
private const val MIN_CROP = 0.05f

/** The band lines in the stretched photo, as fractions of its height. */
fun bandLines(p: EditParams): Pair<Float, Float> {
    val q = p.stretch / 100
    val k = 1 + (p.bandBottom - p.bandTop) * q
    return p.bandTop / k to (p.bandTop + (p.bandBottom - p.bandTop) * (1 + q)) / k
}

/** Inverse of [bandLines]: puts one line at [at] (fraction of the resized photo) and keeps the other. */
fun moveBandLine(p: EditParams, top: Boolean, at: Float): EditParams {
    val q = p.stretch / 100
    return if (top) {
        p.copy(bandTop = (at * (1 + p.bandBottom * q) / (1 + at * q)).coerceIn(0f, p.bandBottom - MIN_BAND))
    } else {
        p.copy(bandBottom = (p.bandTop + (at - p.bandTop) / (1 + q - at * q)).coerceIn(p.bandTop + MIN_BAND, 1f))
    }
}

/** Drags a crop corner (0 top-left, clockwise to 3 bottom-left) or the whole box (4) by a fraction of the photo. */
fun moveCrop(p: EditParams, handle: Int, dx: Float, dy: Float): EditParams {
    fun left() = (p.cropL + dx).coerceIn(0f, p.cropR - MIN_CROP)
    fun right() = (p.cropR + dx).coerceIn(p.cropL + MIN_CROP, 1f)
    fun top() = (p.cropT + dy).coerceIn(0f, p.cropB - MIN_CROP)
    fun bottom() = (p.cropB + dy).coerceIn(p.cropT + MIN_CROP, 1f)
    return when (handle) {
        0 -> p.copy(cropL = left(), cropT = top())
        1 -> p.copy(cropR = right(), cropT = top())
        2 -> p.copy(cropR = right(), cropB = bottom())
        3 -> p.copy(cropL = left(), cropB = bottom())
        4 -> {
            val x = dx.coerceIn(-p.cropL, 1 - p.cropR)
            val y = dy.coerceIn(-p.cropT, 1 - p.cropB)
            p.copy(cropL = p.cropL + x, cropR = p.cropR + x, cropT = p.cropT + y, cropB = p.cropB + y)
        }
        else -> p
    }
}

private fun clamp(v: Float) = if (v <= 0f) 0 else if (v >= 255f) 255 else (v + 0.5f).toInt()

private fun rgb(r: Float, g: Float, b: Float) = (clamp(r) shl 16) or (clamp(g) shl 8) or clamp(b)

private fun luma(r: Int, g: Int, b: Int) = (r * 299 + g * 587 + b * 114) / 1000

private inline fun IntArray.mapRgb(block: (Int, Int, Int) -> Int) {
    for (i in indices) {
        val c = this[i]
        this[i] = (c and 0xFF000000.toInt()) or block(c shr 16 and 255, c shr 8 and 255, c and 255)
    }
}

/** The PC editor's color steps (PIL ImageEnhance semantics, same order) on ARGB pixels, in place. Alpha is kept. */
fun adjustPixels(px: IntArray, w: Int, h: Int, p: EditParams) {
    if (p.brightness != 1f) {
        val f = p.brightness
        px.mapRgb { r, g, b -> rgb(r * f, g * f, b * f) }
    }
    if (p.contrast != 1f) {
        val f = p.contrast
        var sum = 0L
        for (c in px) sum += luma(c shr 16 and 255, c shr 8 and 255, c and 255)
        val mean = (sum.toFloat() / px.size + 0.5f).toInt().toFloat()
        px.mapRgb { r, g, b -> rgb(mean + (r - mean) * f, mean + (g - mean) * f, mean + (b - mean) * f) }
    }
    if (p.saturation != 1f) {
        val f = p.saturation
        px.mapRgb { r, g, b ->
            val l = luma(r, g, b).toFloat()
            rgb(l + (r - l) * f, l + (g - l) * f, l + (b - l) * f)
        }
    }
    if (p.gamma != 1f) {
        val lut = IntArray(256) { (255 * (it / 255f).pow(1 / p.gamma)).roundToInt() }
        px.mapRgb { r, g, b -> (lut[r] shl 16) or (lut[g] shl 8) or lut[b] }
    }
    if (p.temperature != 0f) {
        val t = p.temperature / 100 * 0.25f // +: warmer (more red, less blue)
        val red = IntArray(256) { min(255, (it * (1 + t)).roundToInt()) }
        val blue = IntArray(256) { min(255, (it * (1 - t)).roundToInt()) }
        px.mapRgb { r, g, b -> (red[r] shl 16) or (g shl 8) or blue[b] }
    }
    if (p.sharpness != 1f) {
        // blend with PIL's SMOOTH kernel (1 1 1 / 1 5 1 / 1 1 1) / 13; the border stays as it is, like PIL
        val f = p.sharpness
        val src = px.copyOf()
        for (y in 1 until h - 1) for (x in 1 until w - 1) {
            val i = y * w + x
            var r = 0
            var g = 0
            var b = 0
            for (dy in -1..1) for (dx in -1..1) {
                val c = src[i + dy * w + dx]
                val k = if (dx == 0 && dy == 0) 5 else 1
                r += k * (c shr 16 and 255)
                g += k * (c shr 8 and 255)
                b += k * (c and 255)
            }
            val c = src[i]
            val sr = r / 13f
            val sg = g / 13f
            val sb = b / 13f
            px[i] = (c and 0xFF000000.toInt()) or
                rgb(sr + ((c shr 16 and 255) - sr) * f, sg + ((c shr 8 and 255) - sg) * f, sb + ((c and 255) - sb) * f)
        }
    }
}

private val EDIT_SUFFIX = Regex("_edit[0-9]*(?=\\.[^.]+$)", RegexOption.IGNORE_CASE)

/** x.jpg -> x_edit.jpg, then x_edit2.jpg...; editing x_edit.jpg saves the next version of x, like the PC viewer. */
fun editName(name: String, ext: String, taken: Set<String>): String {
    val stem = EDIT_SUFFIX.replace(name, "").substringBeforeLast(".")
    val lower = taken.map { it.lowercase() }.toSet()
    return (1..999).map { "${stem}_edit${if (it == 1) "" else it}.$ext" }.first { it.lowercase() !in lower }
}

/** Decoded upright (EXIF applied), scaled down to at most [maxPixels]. */
fun decode(context: Context, uri: Uri, maxPixels: Int): Bitmap =
    ImageDecoder.decodeBitmap(ImageDecoder.createSource(context.contentResolver, uri)) { decoder, info, _ ->
        decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
        val (w, h) = info.size.width to info.size.height
        val scale = sqrt(maxPixels / (w.toFloat() * h))
        if (scale < 1) decoder.setTargetSize(max(1, (w * scale).toInt()), max(1, (h * scale).toInt()))
    }

/** Resamples the band between the two lines vertically (legs, torso, anything the user marks), background and all, so straight lines stay straight. */
private fun stretchBand(src: Bitmap, p: EditParams): Bitmap {
    val (w, h) = src.width to src.height
    val top = (p.bandTop * h).roundToInt()
    val bottom = (p.bandBottom * h).roundToInt()
    val band = max(1, ((bottom - top) * (1 + p.stretch / 100)).roundToInt())
    val out = Bitmap.createBitmap(w, h - (bottom - top) + band, Bitmap.Config.ARGB_8888)
    Canvas(out).apply {
        drawBitmap(src, Rect(0, 0, w, top), Rect(0, 0, w, top), null)
        drawBitmap(src, Rect(0, top, w, bottom), Rect(0, top, w, top + band), Paint(Paint.FILTER_BITMAP_FLAG))
        drawBitmap(src, Rect(0, bottom, w, h), Rect(0, top + band, w, out.height), null)
    }
    return out
}

fun render(src: Bitmap, p: EditParams, crop: Boolean = true): Bitmap {
    var bmp = src
    if (p.rot != 0 || p.flip) {
        // mirror first, then turn clockwise, like effects.orient on the PC
        val m = Matrix().apply {
            if (p.flip) postScale(-1f, 1f)
            postRotate(p.rot.toFloat())
        }
        bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
    }
    if (p.stretch != 0f) bmp = stretchBand(bmp, p)
    if (crop && p.cropped) {
        val x = (p.cropL * bmp.width).toInt()
        val y = (p.cropT * bmp.height).toInt()
        bmp = Bitmap.createBitmap(
            bmp, x, y,
            max(1, (p.cropR * bmp.width).toInt() - x), max(1, (p.cropB * bmp.height).toInt() - y),
        )
    }
    if (p.adjusted) {
        val px = IntArray(bmp.width * bmp.height)
        bmp.getPixels(px, 0, bmp.width, 0, 0, bmp.width, bmp.height)
        adjustPixels(px, bmp.width, bmp.height, p)
        bmp = Bitmap.createBitmap(px, bmp.width, bmp.height, Bitmap.Config.ARGB_8888)
    }
    return bmp
}

// ponytail: bigger photos are saved scaled down to this many pixels; tile the pipeline if 50MP originals matter
private const val MAX_SAVE_PIXELS = 24_000_000

/** Saves an edited copy next to the original (which is never changed) and returns it. */
fun saveEdited(context: Context, photo: Photo, folderPath: String, p: EditParams): Photo {
    val bmp = render(decode(context, photo.uri, MAX_SAVE_PIXELS), p)
    val png = photo.name.endsWith(".png", ignoreCase = true)
    val resolver = context.contentResolver
    val taken = mutableSetOf<String>()
    resolver.query(Media.EXTERNAL_CONTENT_URI, arrayOf(Media.DISPLAY_NAME), "${Media.RELATIVE_PATH} = ?", arrayOf(folderPath), null)
        ?.use { while (it.moveToNext()) taken += it.getString(0) }
    val name = editName(photo.name, if (png) "png" else "jpg", taken)
    val uri = resolver.insert(Media.EXTERNAL_CONTENT_URI, ContentValues().apply {
        put(Media.DISPLAY_NAME, name)
        put(Media.MIME_TYPE, if (png) "image/png" else "image/jpeg")
        put(Media.RELATIVE_PATH, folderPath)
        put(Media.DATE_TAKEN, photo.date) // sorts next to the original
        put(Media.IS_PENDING, 1)
    }) ?: error("저장할 파일을 만들지 못했어요")
    try {
        resolver.openOutputStream(uri)!!.use { bmp.compress(if (png) Bitmap.CompressFormat.PNG else Bitmap.CompressFormat.JPEG, 95, it) }
        resolver.update(uri, ContentValues().apply { put(Media.IS_PENDING, 0) }, null, null)
    } catch (e: Throwable) {
        resolver.delete(uri, null, null) // no half-written file left behind
        throw e
    }
    return Photo(ContentUris.parseId(uri), name, photo.date, 0, bmp.width, bmp.height)
}
