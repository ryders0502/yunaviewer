package io.github.ryders0502.yunaviewer

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.os.IBinder
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.cancel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.security.MessageDigest
import kotlin.math.min

/**
 * AI descriptions (outfit, hair, pose, framing) keyed by the file's sha256, like the PC tags.json,
 * so a renamed or moved photo keeps its description. Hashes are cached per MediaStore id.
 */
class Tags private constructor(context: Context) : SQLiteOpenHelper(context, "tags.db", null, 2) {
    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("CREATE TABLE tags (hash TEXT PRIMARY KEY, json TEXT NOT NULL)")
        db.execSQL("CREATE TABLE hashes (id INTEGER PRIMARY KEY, size INTEGER NOT NULL, date INTEGER NOT NULL, hash TEXT NOT NULL)")
        onUpgrade(db, 1, 2)
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        // outfits the user grouped by hand, with the name they gave
        if (oldVersion < 2) db.execSQL("CREATE TABLE outfits (hash TEXT PRIMARY KEY, name TEXT NOT NULL)")
    }

    fun manualOutfit(hash: String): String? =
        readableDatabase.rawQuery("SELECT name FROM outfits WHERE hash = ?", arrayOf(hash)).use {
            if (it.moveToFirst()) it.getString(0) else null
        }

    fun setManualOutfit(hashes: Collection<String>, name: String) = writableDatabase.run {
        beginTransaction()
        try {
            for (hash in hashes) {
                insertWithOnConflict("outfits", null, ContentValues().apply {
                    put("hash", hash)
                    put("name", name)
                }, SQLiteDatabase.CONFLICT_REPLACE)
            }
            setTransactionSuccessful()
        } finally {
            endTransaction()
        }
    }

    /** Hash only if already known for this exact file version; never reads the file. */
    fun cachedHash(photo: Photo): String? =
        readableDatabase.rawQuery("SELECT hash FROM hashes WHERE id = ? AND size = ? AND date = ?",
            arrayOf(photo.id.toString(), photo.size.toString(), photo.date.toString())).use { if (it.moveToFirst()) it.getString(0) else null }

    // ponytail: size + date as the change check; an in-place rewrite with the same size and date keeps the old hash
    fun hash(context: Context, photo: Photo): String = cachedHash(photo) ?: run {
        val digest = MessageDigest.getInstance("SHA-256")
        context.contentResolver.openInputStream(photo.uri)!!.use { input ->
            val buffer = ByteArray(1 shl 16)
            while (true) {
                val n = input.read(buffer)
                if (n < 0) break
                digest.update(buffer, 0, n)
            }
        }
        val hash = digest.digest().joinToString("") { "%02x".format(it) }
        writableDatabase.insertWithOnConflict("hashes", null, ContentValues().apply {
            put("id", photo.id)
            put("size", photo.size)
            put("date", photo.date)
            put("hash", hash)
        }, SQLiteDatabase.CONFLICT_REPLACE)
        hash
    }

    operator fun get(hash: String): JSONObject? =
        readableDatabase.rawQuery("SELECT json FROM tags WHERE hash = ?", arrayOf(hash)).use {
            if (it.moveToFirst()) JSONObject(it.getString(0)) else null
        }

    fun put(hash: String, tags: JSONObject) {
        writableDatabase.insertWithOnConflict("tags", null, ContentValues().apply {
            put("hash", hash)
            put("json", tags.toString())
        }, SQLiteDatabase.CONFLICT_REPLACE)
    }

    /** Photos without a description yet. Reads (hashes) files not seen before, so call it off the main thread. */
    fun missing(context: Context, photos: List<Photo>) = photos.filter { get(hash(context, it)) == null }

    companion object {
        @Volatile private var instance: Tags? = null
        fun of(context: Context) = instance ?: synchronized(this) {
            instance ?: Tags(context.applicationContext).also { instance = it }
        }
    }
}

/** One line for the viewer, e.g. "upperbody · light blue shirt, black skirt · sitting on chair". */
fun tagSummary(tags: JSONObject): String {
    fun shown(key: String) = tags.optString(key).takeUnless { it.isBlank() || it == "none" || it.startsWith("not visible") }
    val clothes = listOf("onepiece", "top", "bottom", "outer").mapNotNull(::shown).joinToString(", ")
    return listOfNotNull(shown("visible"), clothes.ifEmpty { null }, shown("pose")).joinToString(" · ")
}

private const val PER_SHEET = 4 // VERIFY_LAYOUT on the PC: 4 per sheet, 2 columns, 480x600 cells
private const val SHEET_COLUMNS = 2
private const val CELL_W = 480
private const val CELL_H = 600
private const val CHUNK = 8 // DESCRIBE_CHUNK_SIZE on the PC
private const val PARALLEL = 2 // parallel_calls on the PC

/** Labeled contact sheets like make_contact_sheets on the PC: IMG-nnn above each photo, whole photo shown. */
fun contactSheets(context: Context, photos: List<Photo>, ids: List<String>): List<ByteArray> {
    val label = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = android.graphics.Color.WHITE; textSize = 18f }
    val border = Paint().apply { color = 0xFF555555.toInt(); style = Paint.Style.STROKE }
    return photos.indices.chunked(PER_SHEET).map { page ->
        val rows = (page.size + SHEET_COLUMNS - 1) / SHEET_COLUMNS
        val sheet = Bitmap.createBitmap(SHEET_COLUMNS * CELL_W, rows * CELL_H, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(sheet).apply { drawColor(0xFF181818.toInt()) }
        page.forEachIndexed { slot, i ->
            val x = (slot % SHEET_COLUMNS) * CELL_W
            val y = (slot / SHEET_COLUMNS) * CELL_H
            val photo = decode(context, photos[i].uri, CELL_W * CELL_H)
            val scale = min((CELL_W - 12f) / photo.width, (CELL_H - 42f) / photo.height)
            val w = (photo.width * scale).toInt()
            val h = (photo.height * scale).toInt()
            val left = x + (CELL_W - w) / 2
            canvas.drawBitmap(photo, null, android.graphics.Rect(left, y + 34, left + w, y + 34 + h), Paint(Paint.FILTER_BITMAP_FLAG))
            canvas.drawRect(x.toFloat(), y.toFloat(), x + CELL_W - 1f, y + CELL_H - 1f, border)
            canvas.drawText(ids[i], x + 8f, y + 7f - label.ascent(), label)
        }
        ByteArrayOutputStream().also { sheet.compress(Bitmap.CompressFormat.JPEG, 88, it) }.toByteArray()
    }
}

/** Progress of the one folder being indexed (or the last one), shown in the grid and the notification. */
data class IndexJob(val bucket: Long, val total: Int, val done: Int, val running: Boolean, val error: String = "")

object Indexer {
    var job by mutableStateOf<IndexJob?>(null)
        internal set

    /** Starts describing the folder's photos that have no description yet; already described ones are skipped. */
    fun start(context: Context, bucket: Long) {
        if (job?.running == true) return
        context.startForegroundService(Intent(context, IndexService::class.java).putExtra("bucket", bucket))
    }

    fun stop(context: Context) = context.stopService(Intent(context, IndexService::class.java))

    /** Describes [photos] (already filtered to missing ones). Saves after every batch, so a stop loses little. */
    internal suspend fun run(context: Context, gemini: Gemini, bucket: Long, photos: List<Photo>, total: Int) {
        val tags = Tags.of(context)
        var done = total - photos.size
        var error = ""
        job = IndexJob(bucket, total, done, running = true)

        suspend fun describe(batch: List<Photo>) {
            val ids = batch.indices.map { "IMG-%03d".format(it + 1) }
            val result = try {
                gemini.describe(contactSheets(context, batch, ids), ids)
            } catch (e: GeminiError) {
                // a crowded sheet sometimes comes back unusable; halves usually work and are saved on their own
                if (e.malformed && batch.size > 1) {
                    describe(batch.subList(0, batch.size / 2))
                    describe(batch.subList(batch.size / 2, batch.size))
                    return
                }
                error = e.message!!
                if (e.quota) throw e
                return
            } catch (e: java.io.IOException) {
                error = "Gemini에 연결하지 못했어요: ${e.message}"
                return
            }
            for (entry in result) {
                // models sometimes answer "7" or "img-007"
                val n = Regex("\\d+").find(entry.optString("id"))?.value?.toIntOrNull() ?: continue
                val photo = batch.getOrNull(n - 1) ?: continue
                entry.remove("id")
                tags.put(tags.hash(context, photo), entry)
                done++
            }
            job = IndexJob(bucket, total, done, running = true, error)
        }

        try {
            for (pair in photos.chunked(CHUNK).chunked(PARALLEL)) {
                coroutineScope { pair.map { async(Dispatchers.IO) { describe(it) } }.awaitAll() }
            }
        } catch (_: GeminiError) {
            // quota: stop now, the next tap resumes with what is left
        } finally {
            job = IndexJob(bucket, total, done, running = false, error)
        }
    }
}

/** Keeps indexing alive while the screen is off or the app is in the background. */
class IndexService : Service() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // bucket ids are hashes and can be negative, so "no folder" is a missing extra, not -1
        val bucket = intent?.takeIf { it.hasExtra("bucket") }?.getLongExtra("bucket", 0)
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel("index", "AI 분석", NotificationManager.IMPORTANCE_LOW))
        fun notification(text: String, progress: IndexJob?) = Notification.Builder(this, "index")
            .setSmallIcon(android.R.drawable.stat_sys_upload)
            .setContentTitle("Yuna Viewer AI 분석")
            .setContentText(text)
            .setOngoing(true)
            .apply { progress?.let { setProgress(it.total, it.done, false) } }
            .build()
        startForeground(1, notification("준비 중", null), ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        if (bucket == null || Indexer.job?.running == true) return START_NOT_STICKY

        Indexer.job = IndexJob(bucket, 0, 0, running = true)
        scope.launch {
            try {
                val photos = loadPhotos(this@IndexService, bucket).sortedByDescending { it.date }
                val missing = Tags.of(this@IndexService).missing(this@IndexService, photos)
                val progress = launch {
                    while (isActive) {
                        Indexer.job?.let { manager.notify(1, notification("${it.done} / ${it.total}", it)) }
                        kotlinx.coroutines.delay(1000)
                    }
                }
                Indexer.run(this@IndexService, Gemini(getSharedPreferences("prefs", MODE_PRIVATE)), bucket, missing, photos.size)
                progress.cancel()
            } catch (e: Exception) {
                Indexer.job = (Indexer.job ?: IndexJob(bucket, 0, 0, false)).copy(running = false, error = e.message ?: e.javaClass.simpleName)
            } finally {
                stopSelf()
            }
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        scope.cancel()
        Indexer.job = Indexer.job?.copy(running = false)
    }
}
