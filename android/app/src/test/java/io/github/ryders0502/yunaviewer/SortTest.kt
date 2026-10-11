package io.github.ryders0502.yunaviewer

import org.junit.Assert.assertEquals
import org.junit.Test

class SortTest {
    private val a = Photo(1, "b.jpg", date = 300, size = 10, width = 100, height = 100)
    private val b = Photo(2, "A.jpg", date = 100, size = 30, width = 400, height = 300)
    private val c = Photo(3, "c.jpg", date = 200, size = 20, width = 200, height = 200)
    private val photos = listOf(a, b, c)
    private val ratings = mapOf(2L to 5, 3L to 5)

    private fun order(sort: Sort) = sortPhotos(photos, sort) { ratings[it] ?: 0 }.map { it.id }

    @Test
    fun sorts() {
        assertEquals(listOf(1L, 3L, 2L), order(Sort(SortKey.DATE, desc = true)))
        assertEquals(listOf(2L, 1L, 3L), order(Sort(SortKey.NAME, desc = false)))
        assertEquals(listOf(2L, 3L, 1L), order(Sort(SortKey.SIZE, desc = true)))
        assertEquals(listOf(1L, 3L, 2L), order(Sort(SortKey.RESOLUTION, desc = false)))
        // equal ratings fall back to newest first
        assertEquals(listOf(3L, 2L, 1L), order(Sort(SortKey.RATING, desc = true)))
    }

    @Test
    fun parsesSavedSort() {
        assertEquals(Sort(SortKey.SIZE, desc = false), Sort.parse(Sort(SortKey.SIZE, desc = false).toString()))
        assertEquals(Sort(), Sort.parse(null))
        assertEquals(Sort(), Sort.parse("garbage"))
    }
}
