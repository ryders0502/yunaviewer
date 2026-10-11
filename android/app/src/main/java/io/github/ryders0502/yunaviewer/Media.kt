package io.github.ryders0502.yunaviewer

import android.annotation.SuppressLint
import android.app.PendingIntent
import android.app.WallpaperManager
import android.media.ExifInterface
import kotlin.math.roundToInt
import android.content.ContentUris
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.MediaStore.Images.Media
import android.provider.MediaStore

data class Photo(val id: Long, val name: String, val date: Long, val size: Long, val width: Int, val height: Int)

/** A MediaStore bucket: one folder on the phone, browsed one at a time. */
data class Folder(val id: Long, val name: String, val path: String, val coverId: Long, val count: Int)

fun imageUri(id: Long): Uri = ContentUris.withAppendedId(Media.EXTERNAL_CONTENT_URI, id)

val Photo.uri: Uri get() = imageUri(id)

enum class SortKey(val label: String) { DATE("날짜"), NAME("이름"), SIZE("크기"), RESOLUTION("해상도"), RATING("별점") }

data class Sort(val key: SortKey = SortKey.DATE, val desc: Boolean = true) {
    override fun toString() = "${key.name}:${if (desc) 1 else 0}"

    companion object {
        fun parse(text: String?) = runCatching {
            val (key, desc) = text!!.split(":")
            Sort(SortKey.valueOf(key), desc == "1")
        }.getOrDefault(Sort())
    }
}

/** Ties fall back to newest first, like the PC viewer. */
fun sortPhotos(photos: List<Photo>, sort: Sort, rating: (Long) -> Int): List<Photo> {
    val byKey: Comparator<Photo> = when (sort.key) {
        SortKey.DATE -> compareBy { it.date }
        SortKey.NAME -> compareBy(String.CASE_INSENSITIVE_ORDER) { it.name }
        SortKey.SIZE -> compareBy { it.size }
        SortKey.RESOLUTION -> compareBy { it.width.toLong() * it.height }
        SortKey.RATING -> compareBy { rating(it.id) }
    }
    return photos.sortedWith((if (sort.desc) byKey.reversed() else byKey).thenByDescending { it.date })
}

/** Folders ordered by their newest photo; the cover is the photo [chosen] names for the folder, else the newest. */
fun loadFolders(context: Context, chosen: (Long) -> Long = { -1 }): List<Folder> {
    val folders = linkedMapOf<Long, Folder>()
    context.contentResolver.query(
        Media.EXTERNAL_CONTENT_URI,
        arrayOf(Media._ID, Media.BUCKET_ID, Media.BUCKET_DISPLAY_NAME, Media.RELATIVE_PATH),
        null, null, "${Media.DATE_MODIFIED} DESC",
    )?.use { c ->
        while (c.moveToNext()) {
            val bucket = c.getLong(1)
            folders[bucket] = folders[bucket]?.let { it.copy(count = it.count + 1, coverId = if (c.getLong(0) == chosen(bucket)) c.getLong(0) else it.coverId) }
                ?: Folder(bucket, c.getString(2) ?: "", c.getString(3) ?: "", c.getLong(0), 1)
        }
    }
    return folders.values.toList()
}

fun loadPhotos(context: Context, bucket: Long): List<Photo> {
    val photos = mutableListOf<Photo>()
    context.contentResolver.query(
        Media.EXTERNAL_CONTENT_URI,
        arrayOf(Media._ID, Media.DISPLAY_NAME, Media.DATE_TAKEN, Media.DATE_MODIFIED, Media.SIZE, Media.WIDTH, Media.HEIGHT, Media.ORIENTATION),
        "${Media.BUCKET_ID} = ?", arrayOf(bucket.toString()), null,
    )?.use { c ->
        while (c.moveToNext()) {
            val taken = c.getLong(2)
            val date = if (taken > 0) taken else c.getLong(3) * 1000
            // MediaStore sizes are as stored; a 90/270 EXIF turn shows the photo the other way up
            val turned = c.getInt(7) % 180 != 0
            val (w, h) = if (turned) c.getInt(6) to c.getInt(5) else c.getInt(5) to c.getInt(6)
            photos += Photo(c.getLong(0), c.getString(1) ?: "", date, c.getLong(4), w, h)
        }
    }
    return photos
}

/** The system trash: the user confirms, and can restore from the stock gallery for 30 days. */
fun trashRequest(context: Context, ids: Collection<Long>): PendingIntent =
    MediaStore.createTrashRequest(context.contentResolver, ids.map(::imageUri), true)

fun writeRequest(context: Context, ids: Collection<Long>): PendingIntent =
    MediaStore.createWriteRequest(context.contentResolver, ids.map(::imageUri))

/** A new folder next to [current] (same parent), as a MediaStore relative path, or null for an unusable name. */
fun siblingPath(current: Folder, name: String): String? {
    val clean = name.trim()
    if (clean.isEmpty() || clean.startsWith(".") || clean.any { it in "/\\:*?\"<>|" }) return null
    val parent = current.path.trimEnd('/').substringBeforeLast('/', "")
    // photos must stay under a media folder such as Pictures/ or DCIM/
    return (if (parent.isEmpty()) "Pictures" else parent) + "/" + clean + "/"
}

/** Moves every photo of [folder] into a sibling folder named [name]; other files stay in the old folder. */
fun renameFolder(context: Context, folder: Folder, name: String): Int {
    val path = siblingPath(folder, name) ?: return -1
    return moveTo(context, loadPhotos(context, folder.id).map { it.id }, path)
}

/** Needs write access from [writeRequest] first. Returns how many could not be moved (e.g. same name already there). */
fun moveTo(context: Context, ids: Collection<Long>, path: String): Int {
    val values = ContentValues().apply { put(Media.RELATIVE_PATH, path) }
    return ids.count { id ->
        runCatching { context.contentResolver.update(imageUri(id), values, null, null) }.getOrDefault(0) == 0
    }
}

/** [which]: WallpaperManager.FLAG_SYSTEM (home), FLAG_LOCK or both; the system crops it to the screen. */
fun setWallpaper(context: Context, photo: Photo, which: Int) {
    context.contentResolver.openInputStream(photo.uri)!!.use {
        WallpaperManager.getInstance(context).setStream(it, null, true, which)
    }
}

/** Label/value rows for the details dialog: file facts plus camera EXIF when the file has it. */
// the framework ExifInterface is fine from API 30 (our minSdk); the androidx copy would be one more dependency
@SuppressLint("ExifInterface")
fun photoDetails(context: Context, photo: Photo): List<Pair<String, String>> {
    val rows = mutableListOf<Pair<String, String>>()
    var path = ""
    var mime = ""
    context.contentResolver.query(photo.uri, arrayOf(Media.RELATIVE_PATH, Media.MIME_TYPE), null, null, null)?.use {
        if (it.moveToFirst()) {
            path = it.getString(0) ?: ""
            mime = it.getString(1) ?: ""
        }
    }
    rows += "파일 이름" to photo.name
    if (path.isNotEmpty()) rows += "위치" to path
    rows += "날짜" to java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.KOREA).format(java.util.Date(photo.date))
    rows += "해상도" to "${photo.width} × ${photo.height} (%.1fMP)".format(photo.width * photo.height / 1e6)
    rows += "파일 크기" to if (photo.size >= 1 shl 20) "%.1f MB".format(photo.size / 1048576.0) else "${photo.size / 1024} KB"
    if (mime.isNotEmpty()) rows += "형식" to mime
    runCatching {
        context.contentResolver.openInputStream(photo.uri)!!.use { input ->
            val exif = ExifInterface(input)
            fun add(label: String, vararg tags: String) {
                tags.firstNotNullOfOrNull { exif.getAttribute(it)?.trim()?.takeIf(String::isNotEmpty) }?.let { rows += label to it }
            }
            val camera = listOfNotNull(exif.getAttribute(ExifInterface.TAG_MAKE), exif.getAttribute(ExifInterface.TAG_MODEL))
                .map(String::trim).filter(String::isNotEmpty).distinct().joinToString(" ")
            if (camera.isNotEmpty()) rows += "카메라" to camera
            add("촬영 일시", ExifInterface.TAG_DATETIME_ORIGINAL, ExifInterface.TAG_DATETIME)
            exif.getAttributeDouble(ExifInterface.TAG_F_NUMBER, 0.0).takeIf { it > 0 }?.let { rows += "조리개" to "f/%.1f".format(it) }
            exif.getAttributeDouble(ExifInterface.TAG_EXPOSURE_TIME, 0.0).takeIf { it > 0 }?.let {
                rows += "셔터" to if (it < 1) "1/${(1 / it).roundToInt()}초" else "%.1f초".format(it)
            }
            add("ISO", ExifInterface.TAG_ISO_SPEED_RATINGS)
            exif.getAttributeDouble(ExifInterface.TAG_FOCAL_LENGTH, 0.0).takeIf { it > 0 }?.let { rows += "초점 거리" to "%.1fmm".format(it) }
            add("소프트웨어", ExifInterface.TAG_SOFTWARE)
            add("설명", ExifInterface.TAG_IMAGE_DESCRIPTION, ExifInterface.TAG_USER_COMMENT)
        }
    }
    return rows
}

fun shareIntent(ids: Collection<Long>): Intent = Intent.createChooser(
    Intent(Intent.ACTION_SEND_MULTIPLE).apply {
        type = "image/*"
        putParcelableArrayListExtra(Intent.EXTRA_STREAM, ArrayList(ids.map(::imageUri)))
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    },
    null,
)
