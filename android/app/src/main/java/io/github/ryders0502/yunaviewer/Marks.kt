package io.github.ryders0502.yunaviewer

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import androidx.compose.runtime.mutableStateMapOf

data class Mark(val fav: Boolean = false, val rating: Int = 0, val note: String = "")

/**
 * Favorites, ratings and notes keyed by MediaStore id, which survives moves between folders.
 * ponytail: all rows held in memory; fine for tens of thousands of marks.
 */
class Marks(context: Context) : SQLiteOpenHelper(context, "marks.db", null, 1) {
    override fun onCreate(db: SQLiteDatabase) =
        db.execSQL("CREATE TABLE marks (id INTEGER PRIMARY KEY, fav INTEGER NOT NULL, rating INTEGER NOT NULL, note TEXT NOT NULL)")

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {}

    private val all = mutableStateMapOf<Long, Mark>().apply {
        readableDatabase.rawQuery("SELECT id, fav, rating, note FROM marks", null).use { c ->
            while (c.moveToNext()) put(c.getLong(0), Mark(c.getInt(1) == 1, c.getInt(2), c.getString(3)))
        }
    }

    operator fun get(id: Long) = all[id] ?: Mark()

    operator fun set(id: Long, mark: Mark) {
        all[id] = mark
        if (mark == Mark()) {
            writableDatabase.delete("marks", "id = ?", arrayOf(id.toString()))
        } else {
            val values = ContentValues().apply {
                put("id", id)
                put("fav", if (mark.fav) 1 else 0)
                put("rating", mark.rating)
                put("note", mark.note)
            }
            writableDatabase.insertWithOnConflict("marks", null, values, SQLiteDatabase.CONFLICT_REPLACE)
        }
    }
}
