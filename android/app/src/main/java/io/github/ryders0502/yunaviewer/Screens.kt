@file:OptIn(ExperimentalMaterial3Api::class)

package io.github.ryders0502.yunaviewer

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.awaitLongPressOrCancellation
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.staggeredgrid.LazyStaggeredGridState
import androidx.compose.foundation.lazy.staggeredgrid.LazyVerticalStaggeredGrid
import androidx.compose.foundation.lazy.staggeredgrid.StaggeredGridCells
import androidx.compose.foundation.lazy.staggeredgrid.items as staggeredItems
import androidx.compose.foundation.interaction.DragInteraction
import androidx.compose.foundation.gestures.snapping.SnapPosition
import androidx.compose.foundation.gestures.snapping.rememberSnapFlingBehavior
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.FavoriteBorder
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.lazy.grid.GridItemSpan
import android.app.WallpaperManager
import android.widget.Toast
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Surface
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shadow
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.IntRect
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.platform.LocalWindowInfo
import androidx.compose.foundation.layout.height
import androidx.compose.ui.unit.round
import androidx.compose.ui.unit.sp
import androidx.compose.runtime.produceState
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import coil3.compose.AsyncImage
import coil3.request.ImageRequest
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

private val Gold = Color(0xFFFFD54D)
private val Pink = Color(0xFFFF6B8B)
private val Scrim = Color(0x99000000)
private val Badge = TextStyle(fontSize = 12.sp, shadow = Shadow(Color.Black, blurRadius = 4f))

@Composable
fun FolderScreen(
    folders: List<Folder>,
    showHidden: Boolean,
    onShowHidden: (Boolean) -> Unit,
    onSettings: () -> Unit,
    onRename: (Folder) -> Unit,
    onCover: (Folder) -> Unit,
    onHide: (Folder, Boolean) -> Unit,
    onOpen: (Folder) -> Unit,
) {
    var menuFor by remember { mutableStateOf<Long?>(null) }
    BackHandler(enabled = showHidden) { onShowHidden(false) }
    Scaffold(topBar = {
        TopAppBar(
            title = { Text(if (showHidden) "숨긴 폴더" else "Yuna Viewer") },
            navigationIcon = { if (showHidden) IconButton({ onShowHidden(false) }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "뒤로") } },
            actions = {
                if (!showHidden) TextButton({ onShowHidden(true) }) { Text("숨긴 폴더") }
                TextButton(onSettings) { Text("설정") }
            },
        )
    }) { pad ->
        LazyVerticalGrid(
            GridCells.Fixed(2),
            Modifier.padding(top = pad.calculateTopPadding()).fillMaxSize(),
            contentPadding = PaddingValues(8.dp, 0.dp, 8.dp, pad.calculateBottomPadding()),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            if (folders.isEmpty()) item(span = { GridItemSpan(maxLineSpan) }) {
                Text(if (showHidden) "숨긴 폴더가 없어요. 폴더를 길게 눌러 숨길 수 있어요." else "사진이 있는 폴더가 없어요.", Modifier.padding(16.dp))
            }
            items(folders, key = { it.id }) { folder ->
                // long press: rename, hide or unhide
                Column(Modifier.combinedClickable(onLongClick = { menuFor = folder.id }) { onOpen(folder) }) {
                    DropdownMenu(menuFor == folder.id, { menuFor = null }) {
                        DropdownMenuItem({ Text("이름 변경") }, {
                            menuFor = null
                            onRename(folder)
                        })
                        DropdownMenuItem({ Text("대표 사진") }, {
                            menuFor = null
                            onCover(folder)
                        })
                        DropdownMenuItem({ Text(if (showHidden) "숨기기 취소" else "숨기기") }, {
                            menuFor = null
                            onHide(folder, !showHidden)
                        })
                    }
                    AsyncImage(
                        imageUri(folder.coverId), null,
                        Modifier.aspectRatio(1f).clip(RoundedCornerShape(8.dp)),
                        contentScale = ContentScale.Crop,
                        // portraits: show the top, where the face is, not the middle
                        alignment = Alignment.TopCenter,
                    )
                    Text(folder.name, maxLines = 1, overflow = TextOverflow.Ellipsis, style = MaterialTheme.typography.bodyLarge)
                    Text("${folder.count}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
    }
}

/** Full-screen list of a folder's photos; tapping one makes it the folder's cover. */
@Composable
fun CoverPicker(folder: Folder, onPick: (Photo) -> Unit, onDismiss: () -> Unit) {
    val context = LocalContext.current
    val photos by produceState(emptyList<Photo>(), folder) { value = withContext(Dispatchers.IO) { loadPhotos(context, folder.id).sortedByDescending { it.date } } }
    Dialog(onDismiss, DialogProperties(usePlatformDefaultWidth = false)) {
        Surface(Modifier.fillMaxSize()) {
            Column {
                Text("${folder.name}: 대표 사진 선택", Modifier.statusBarsPadding().padding(16.dp), style = MaterialTheme.typography.titleMedium)
                LazyVerticalGrid(GridCells.Fixed(3), Modifier.weight(1f), horizontalArrangement = Arrangement.spacedBy(2.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    items(photos, key = { it.id }) { p ->
                        AsyncImage(p.uri, null, Modifier.aspectRatio(1f).clickable { onPick(p) }, contentScale = ContentScale.Crop, alignment = Alignment.TopCenter)
                    }
                }
                TextButton(onDismiss, Modifier.align(Alignment.End).navigationBarsPadding()) { Text("취소") }
            }
        }
    }
}

private fun LazyStaggeredGridState.indexAt(position: Offset) =
    layoutInfo.visibleItemsInfo.firstOrNull { IntRect(it.offset, it.size).contains(position.round()) }?.index

@Composable
fun GridScreen(
    title: String,
    photos: List<Photo>,
    badges: Map<Long, Int>,
    banner: @Composable () -> Unit,
    marks: Marks,
    state: LazyStaggeredGridState,
    columns: Int,
    onColumns: (Int) -> Unit,
    keepRatio: Boolean,
    onKeepRatio: (Boolean) -> Unit,
    sort: Sort,
    onSort: (Sort) -> Unit,
    outfitMode: Boolean,
    onOutfitMode: (Boolean) -> Unit,
    selected: Set<Long>,
    partial: Set<Long>,
    onUp: (() -> Unit)?,
    onTitleLongClick: (() -> Unit)?,
    onSelect: (Set<Long>) -> Unit,
    onOpen: (Int) -> Unit,
    onBack: () -> Unit,
    onShare: () -> Unit,
    onAsk: () -> Unit,
    onGroup: () -> Unit,
    onMove: () -> Unit,
    onTrash: () -> Unit,
) {
    // gesture handlers outlive recompositions, so they read the latest values through these
    val currentPhotos by rememberUpdatedState(photos)
    val currentSelected by rememberUpdatedState(selected)
    val currentColumns by rememberUpdatedState(columns)
    val open by rememberUpdatedState(onOpen)
    val select by rememberUpdatedState(onSelect)
    val setColumns by rememberUpdatedState(onColumns)
    val haptic = LocalHapticFeedback.current
    // one tile width for every column: lanes differ by a pixel, and the grid fills the shortest lane first,
    // so heights from each lane's own width would scatter equal photos across rows
    val tileWidth = (LocalWindowInfo.current.containerDpSize.width - 2.dp * (columns - 1)) / columns

    Scaffold(topBar = { Column {
        if (selected.isEmpty()) {
            TopAppBar(
                // inside an outfit, long press on the title renames the outfit
                title = {
                    Text(
                        title, Modifier.combinedClickable(enabled = onTitleLongClick != null, onClick = {}, onLongClick = onTitleLongClick),
                        maxLines = 1, overflow = TextOverflow.Ellipsis,
                    )
                },
                navigationIcon = { IconButton(onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "뒤로") } },
                actions = { SortMenu(sort, onSort, keepRatio, onKeepRatio, outfitMode, onOutfitMode) },
            )
        } else {
            TopAppBar(
                title = { Text("${selected.size}") },
                navigationIcon = { IconButton({ onSelect(emptySet()) }) { Icon(Icons.Filled.Close, "선택 해제") } },
                actions = {
                    if (onUp != null) IconButton(onUp) { Icon(painterResource(R.drawable.ic_arrow_upward), "상위로 (선택 유지)") }
                    IconButton({ onSelect(photos.map { it.id }.toSet()) }) { Icon(painterResource(R.drawable.ic_select_all), "전체 선택") }
                    IconButton(onMove) { Icon(painterResource(R.drawable.ic_drive_file_move), "폴더로 이동") }
                    IconButton(onShare) { Icon(Icons.Filled.Share, "공유") }
                    IconButton(onTrash) { Icon(Icons.Filled.Delete, "휴지통") }
                    var more by remember { mutableStateOf(false) }
                    Box {
                        IconButton({ more = true }) { Icon(Icons.Filled.MoreVert, "더 보기") }
                        DropdownMenu(more, { more = false }) {
                            DropdownMenuItem({ Text("같은 의상으로 묶기") }, {
                                more = false
                                onGroup()
                            })
                            DropdownMenuItem({ Text("AI에게 질문 (${MAX_ASK_IMAGES}장까지)") }, {
                                more = false
                                onAsk()
                            }, enabled = selected.size <= MAX_ASK_IMAGES)
                        }
                    }
                },
            )
        }
        banner()
    } }) { pad ->
        // staggered like the PC masonry: each photo goes into the shortest column; square tiles make it a plain grid
        LazyVerticalStaggeredGrid(
            StaggeredGridCells.Fixed(columns),
            Modifier
                .padding(top = pad.calculateTopPadding())
                .fillMaxSize()
                // pinch changes the column count; runs before the grid's own scrolling sees the fingers
                .pointerInput(Unit) {
                    awaitEachGesture {
                        awaitFirstDown(requireUnconsumed = false)
                        var zoom = 1f
                        do {
                            val event = awaitPointerEvent(PointerEventPass.Initial)
                            if (event.changes.count { it.pressed } >= 2) {
                                zoom *= event.calculateZoom()
                                val step = if (zoom > 1.25f) -1 else if (zoom < 0.8f) 1 else 0
                                if (step != 0) {
                                    zoom = 1f
                                    val next = (currentColumns + step).coerceIn(2, 8)
                                    if (next != currentColumns) setColumns(next)
                                }
                                event.changes.forEach { it.consume() }
                            }
                        } while (event.changes.any { it.pressed })
                    }
                }
                // long press selects, then dragging extends the selection over a range
                // ponytail: no auto-scroll at the screen edge while drag-selecting
                .pointerInput(Unit) {
                    awaitEachGesture {
                        val down = awaitFirstDown(requireUnconsumed = false)
                        val press = awaitLongPressOrCancellation(down.id) ?: return@awaitEachGesture
                        val anchor = state.indexAt(press.position) ?: return@awaitEachGesture
                        haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                        val base = currentSelected
                        val adding = currentPhotos[anchor].id !in base
                        fun selectTo(index: Int) {
                            val range = currentPhotos.subList(min(anchor, index), max(anchor, index) + 1).map { it.id }
                            select(if (adding) base + range else base - range.toSet())
                        }
                        selectTo(anchor)
                        while (true) {
                            val change = awaitPointerEvent(PointerEventPass.Initial).changes
                                .firstOrNull { it.id == press.id }
                            if (change == null || !change.pressed) break
                            change.consume()
                            state.indexAt(change.position)?.let(::selectTo)
                        }
                    }
                }
                .pointerInput(Unit) {
                    // the no-op long press keeps a held finger from also counting as a tap
                    detectTapGestures(onLongPress = {}) { position ->
                        val index = state.indexAt(position) ?: return@detectTapGestures
                        val id = currentPhotos[index].id
                        when {
                            currentSelected.isEmpty() -> open(index)
                            id in currentSelected -> select(currentSelected - id)
                            else -> select(currentSelected + id)
                        }
                    }
                },
            state = state,
            contentPadding = PaddingValues(bottom = pad.calculateBottomPadding()),
            horizontalArrangement = Arrangement.spacedBy(2.dp),
            verticalItemSpacing = 2.dp,
        ) {
            staggeredItems(photos, key = { it.id }) { photo ->
                Tile(photo, marks[photo.id], photo.id in selected, photo.id in partial, keepRatio, badges[photo.id], tileWidth)
            }
        }
    }
}

@Composable
private fun SortMenu(
    sort: Sort,
    onSort: (Sort) -> Unit,
    keepRatio: Boolean,
    onKeepRatio: (Boolean) -> Unit,
    outfitMode: Boolean,
    onOutfitMode: (Boolean) -> Unit,
) {
    var open by remember { mutableStateOf(false) }
    Box {
        TextButton({ open = true }) { Text(sort.key.label + if (sort.desc) " ↓" else " ↑") }
        DropdownMenu(open, { open = false }) {
            SortKey.entries.forEach { key ->
                DropdownMenuItem(
                    text = { Text(key.label + if (key == sort.key) if (sort.desc) "  ↓" else "  ↑" else "") },
                    onClick = {
                        open = false
                        // picking the current key again flips the direction
                        onSort(if (key == sort.key) sort.copy(desc = !sort.desc) else Sort(key, desc = key != SortKey.NAME))
                    },
                )
            }
            HorizontalDivider()
            DropdownMenuItem(
                text = { Text("같은 의상 묶어 보기" + if (outfitMode) "  ✓" else "") },
                onClick = {
                    open = false
                    onOutfitMode(!outfitMode)
                },
            )
            HorizontalDivider()
            for ((ratio, label) in listOf(true to "썸네일: 원본 비율", false to "썸네일: 정사각형")) {
                DropdownMenuItem(
                    text = { Text(label + if (ratio == keepRatio) "  ✓" else "") },
                    onClick = {
                        open = false
                        onKeepRatio(ratio)
                    },
                )
            }
        }
    }
}

@Composable
private fun Tile(photo: Photo, mark: Mark, selected: Boolean, partial: Boolean, keepRatio: Boolean, outfitCount: Int?, width: Dp) {
    // whole photo, nothing cut off; square crops show more photos per screen
    val ratio = if (keepRatio && photo.width > 0 && photo.height > 0) photo.width / photo.height.toFloat() else 1f
    Box(Modifier.fillMaxWidth().height(width / ratio.coerceIn(0.25f, 4f))) {
        AsyncImage(
            photo.uri, null,
            if (selected) Modifier.fillMaxSize().padding(8.dp).clip(RoundedCornerShape(6.dp)) else Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop,
        )
        if (mark.fav) Text("♥", Modifier.align(Alignment.TopEnd).padding(4.dp), color = Pink, style = Badge)
        if (mark.rating > 0) Text("★${mark.rating}", Modifier.align(Alignment.BottomStart).padding(4.dp), color = Gold, style = Badge)
        if (mark.note.isNotEmpty()) Text("✎", Modifier.align(Alignment.BottomEnd).padding(4.dp), color = Color.White, style = Badge)
        if (outfitCount != null) {
            Text(
                "👗 ${outfitCount}장",
                Modifier.align(Alignment.BottomCenter).padding(4.dp).background(Scrim, RoundedCornerShape(999.dp)).padding(horizontal = 6.dp),
                color = Color.White, style = Badge,
            )
        }
        if (selected) {
            Icon(Icons.Filled.CheckCircle, "선택됨", Modifier.align(Alignment.TopStart).padding(2.dp), tint = MaterialTheme.colorScheme.primary)
        } else if (partial) {
            // some photos of this outfit are selected, not all
            Icon(Icons.Filled.CheckCircle, "일부 선택됨", Modifier.align(Alignment.TopStart).padding(2.dp), tint = MaterialTheme.colorScheme.primary.copy(alpha = 0.45f))
        }
    }
}

@Composable
fun ViewerScreen(
    photos: List<Photo>,
    start: Int,
    marks: Marks,
    describe: (Photo) -> String?,
    onClose: (Int) -> Unit,
    onEdit: (Photo) -> Unit,
    onShare: (Photo) -> Unit,
    onTrash: (Photo) -> Unit,
) {
    val pager = rememberPagerState(start) { photos.size }
    val page = pager.currentPage.coerceAtMost(photos.lastIndex)
    val photo = photos[page]
    val mark = marks[photo.id]
    // the filmstrip is endless: thumbnail i shows photo i % n, so the strip wraps around the first/last photo
    val n = photos.size
    val mid = Int.MAX_VALUE / 2 / n * n
    val strip = rememberLazyListState(mid + start)
    val scope = rememberCoroutineScope()
    var chrome by remember { mutableStateOf(true) }
    var editingNote by remember { mutableStateOf(false) }
    var wallpaperMenu by remember { mutableStateOf(false) }
    var details by remember { mutableStateOf<List<Pair<String, String>>?>(null) }
    val context = LocalContext.current

    BackHandler { onClose(page) }
    fun wallpaper(which: Int, label: String) {
        wallpaperMenu = false
        scope.launch {
            val ok = withContext(Dispatchers.IO) { runCatching { setWallpaper(context, photo, which) } }
            Toast.makeText(context, ok.fold({ "$label 배경으로 설정했어요" }, { "배경 설정 실패: ${it.message}" }), Toast.LENGTH_SHORT).show()
        }
    }
    // the thumbnail in the middle of the strip is the current photo
    val center by remember {
        derivedStateOf {
            val info = strip.layoutInfo
            // item offsets are measured from the start of the content, i.e. after the padding
            val middle = (info.viewportStartOffset + info.viewportEndOffset) / 2
            info.visibleItemsInfo.minByOrNull { abs(it.offset + it.size / 2 - middle) }?.index ?: (mid + start)
        }
    }
    // padding of half the strip puts the first/last item in the middle; scrolling to an item then centers it
    val sidePad = (LocalConfiguration.current.screenWidthDp.dp - 52.dp) / 2
    suspend fun centerOn(index: Int, animate: Boolean) {
        if (animate) strip.animateScrollToItem(index) else strip.scrollToItem(index)
    }
    // only a touch (drag, fling, tap on a thumbnail) moves the big photo along; centering the strip by code must not
    var follow by remember { mutableStateOf(false) }
    LaunchedEffect(chrome) {
        if (!chrome) return@LaunchedEffect
        launch { strip.interactionSource.interactions.collect { if (it is DragInteraction.Start) follow = true } }
        launch { snapshotFlow { strip.isScrollInProgress }.collect { if (!it) follow = false } }
        snapshotFlow { center }.collect { c ->
            if (follow && c % n != pager.currentPage) pager.scrollToPage(c % n)
        }
    }
    // swiping the big photo moves the strip, by the shorter way round
    LaunchedEffect(page, chrome) {
        if (!chrome || follow) return@LaunchedEffect
        val from = center
        val step = ((page - from % n) % n + n + n / 2) % n - n / 2
        centerOn(from + step, animate = from != mid + start)
    }

    CompositionLocalProvider(LocalContentColor provides Color.White) {
        Box(Modifier.fillMaxSize().background(Color.Black)) {
            HorizontalPager(pager, Modifier.fillMaxSize(), key = { photos[it].id }) { index ->
                ZoomableImage(photos[index], onTap = { chrome = !chrome }, onLongPress = {
                    scope.launch { details = withContext(Dispatchers.IO) { photoDetails(context, photos[index]) } }
                }, onDismiss = { onClose(pager.currentPage) })
            }
            if (chrome) {
                Row(
                    Modifier.align(Alignment.TopCenter).fillMaxWidth().background(Scrim).statusBarsPadding(),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    IconButton({ onClose(page) }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "닫기") }
                    Column(Modifier.weight(1f)) {
                        Text(photo.name, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        Text("${page + 1} / ${photos.size} · ${photo.width}×${photo.height}", style = MaterialTheme.typography.bodySmall)
                    }
                }
                Column(Modifier.align(Alignment.BottomCenter).fillMaxWidth().background(Scrim).navigationBarsPadding()) {
                    if (mark.note.isNotEmpty()) Text(mark.note, Modifier.padding(12.dp, 8.dp, 12.dp, 0.dp))
                    describe(photo)?.let {
                        Text(it, Modifier.padding(12.dp, 6.dp, 12.dp, 0.dp), style = MaterialTheme.typography.bodySmall, maxLines = 2, overflow = TextOverflow.Ellipsis)
                    }
                    LazyRow(
                        state = strip,
                        flingBehavior = rememberSnapFlingBehavior(strip, SnapPosition.Center),
                        contentPadding = PaddingValues(horizontal = sidePad, vertical = 8.dp),
                        horizontalArrangement = Arrangement.spacedBy(3.dp),
                    ) {
                        items(Int.MAX_VALUE) { index ->
                            val shape = RoundedCornerShape(4.dp)
                            AsyncImage(
                                photos[index % n].uri, null,
                                Modifier.size(52.dp).clip(shape)
                                    .then(if (index == center) Modifier.border(2.dp, Color.White, shape) else Modifier.alpha(0.55f))
                                    .clickable { scope.launch { follow = true; centerOn(index, animate = true); follow = false } },
                                contentScale = ContentScale.Crop,
                            )
                        }
                    }
                    Row(Modifier.fillMaxWidth().padding(horizontal = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                        IconButton({ marks[photo.id] = mark.copy(fav = !mark.fav) }) {
                            Icon(if (mark.fav) Icons.Filled.Favorite else Icons.Filled.FavoriteBorder, "즐겨찾기", tint = if (mark.fav) Pink else Color.White)
                        }
                        (1..5).forEach { n ->
                            Text(
                                if (n <= mark.rating) "★" else "☆",
                                Modifier.clickable { marks[photo.id] = mark.copy(rating = if (mark.rating == n) 0 else n) }.padding(3.dp),
                                color = Gold, fontSize = 22.sp,
                            )
                        }
                        // the five actions share the rest of the row equally
                        val slot = Modifier.weight(1f)
                        IconButton({ editingNote = true }, slot) {
                            Icon(painterResource(R.drawable.ic_note), "메모", tint = if (mark.note.isNotEmpty()) Gold else Color.White)
                        }
                        IconButton({ onEdit(photo) }, slot) { Icon(Icons.Filled.Edit, "편집") }
                        Box(slot, contentAlignment = Alignment.Center) {
                            IconButton({ wallpaperMenu = true }) { Icon(painterResource(R.drawable.ic_wallpaper), "배경 화면으로 설정") }
                            DropdownMenu(wallpaperMenu, { wallpaperMenu = false }) {
                                DropdownMenuItem({ Text("홈 화면") }, { wallpaper(WallpaperManager.FLAG_SYSTEM, "홈 화면") })
                                DropdownMenuItem({ Text("잠금 화면") }, { wallpaper(WallpaperManager.FLAG_LOCK, "잠금 화면") })
                                DropdownMenuItem({ Text("홈 화면과 잠금 화면") }, {
                                    wallpaper(WallpaperManager.FLAG_SYSTEM or WallpaperManager.FLAG_LOCK, "홈·잠금 화면")
                                })
                            }
                        }
                        IconButton({ onShare(photo) }, slot) { Icon(Icons.Filled.Share, "공유") }
                        IconButton({ onTrash(photo) }, slot) { Icon(Icons.Filled.Delete, "휴지통") }
                    }
                }
            }
        }
    }

    details?.let { rows ->
        AlertDialog(
            onDismissRequest = { details = null },
            confirmButton = { TextButton({ details = null }) { Text("닫기") } },
            title = { Text("상세 정보") },
            text = {
                SelectionContainer {
                    Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        for ((label, value) in rows) {
                            Column {
                                Text(label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                Text(value)
                            }
                        }
                    }
                }
            },
        )
    }

    if (editingNote) {
        var text by remember { mutableStateOf(mark.note) }
        AlertDialog(
            onDismissRequest = { editingNote = false },
            confirmButton = {
                TextButton({
                    marks[photo.id] = mark.copy(note = text.trim())
                    editingNote = false
                }) { Text("저장") }
            },
            dismissButton = { TextButton({ editingNote = false }) { Text("취소") } },
            title = { Text("메모") },
            text = { OutlinedTextField(text, { text = it }) },
        )
    }
}

private const val MAX_ZOOM = 5f

/** Pinch or double tap to zoom; one finger pans only while zoomed, otherwise the pager swipes. Swipe down closes. */
@Composable
private fun ZoomableImage(photo: Photo, onTap: () -> Unit, onLongPress: () -> Unit, onDismiss: () -> Unit) {
    var scale by remember { mutableFloatStateOf(1f) }
    var offset by remember { mutableStateOf(Offset.Zero) }
    val dismissPx = with(LocalDensity.current) { 120.dp.toPx() }
    val tap by rememberUpdatedState(onTap)
    val longPress by rememberUpdatedState(onLongPress)
    val dismiss by rememberUpdatedState(onDismiss)
    fun clamp(o: Offset, size: IntSize): Offset {
        val maxX = (scale - 1) * size.width / 2
        val maxY = (scale - 1) * size.height / 2
        return Offset(o.x.coerceIn(-maxX, maxX), o.y.coerceIn(-maxY, maxY))
    }

    AsyncImage(
        // the PC viewer also caps its view image at 2400px; full resolution would risk running out of memory
        ImageRequest.Builder(LocalContext.current).data(photo.uri).size(2560).build(),
        photo.name,
        Modifier
            .fillMaxSize()
            .pointerInput(Unit) {
                detectTapGestures(
                    onTap = { tap() },
                    onLongPress = { longPress() },
                    onDoubleTap = { at ->
                        if (scale > 1f) {
                            scale = 1f
                            offset = Offset.Zero
                        } else {
                            scale = 2.5f
                            offset = clamp((Offset(size.width / 2f, size.height / 2f) - at) * (scale - 1), size)
                        }
                    },
                )
            }
            .pointerInput(Unit) {
                awaitEachGesture {
                    awaitFirstDown(requireUnconsumed = false)
                    var drag = Offset.Zero
                    do {
                        val event = awaitPointerEvent()
                        val zoom = event.calculateZoom()
                        val pan = event.calculatePan()
                        if (event.changes.count { it.pressed } <= 1 && scale == 1f) {
                            drag += pan
                        } else if (zoom != 1f || pan != Offset.Zero) {
                            // a still finger stays unconsumed so a tap still toggles the bars
                            scale = (scale * zoom).coerceIn(1f, MAX_ZOOM)
                            offset = clamp(offset + pan, size)
                            event.changes.forEach { it.consume() }
                        }
                    } while (event.changes.any { it.pressed })
                    if (scale == 1f && drag.y > dismissPx && drag.y > 2 * abs(drag.x)) dismiss()
                }
            }
            .graphicsLayer {
                scaleX = scale
                scaleY = scale
                translationX = offset.x
                translationY = offset.y
            },
        contentScale = ContentScale.Fit,
    )
}
