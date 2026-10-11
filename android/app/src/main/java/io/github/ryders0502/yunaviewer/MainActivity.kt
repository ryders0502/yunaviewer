package io.github.ryders0502.yunaviewer

import android.Manifest
import android.app.Activity
import android.app.PendingIntent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.widget.Toast
import android.graphics.Color
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.OutlinedTextField
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.staggeredgrid.LazyStaggeredGridState
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.edit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // the app is always dark, so keep the bar icons light even when the phone is in light mode
        enableEdgeToEdge(SystemBarStyle.dark(Color.TRANSPARENT), SystemBarStyle.dark(Color.TRANSPARENT))
        val marks = Marks(this)
        val prefs = getSharedPreferences("prefs", MODE_PRIVATE)
        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                Surface { App(marks, prefs) }
            }
        }
    }
}

private val READ_PERMISSION =
    if (Build.VERSION.SDK_INT >= 33) Manifest.permission.READ_MEDIA_IMAGES else Manifest.permission.READ_EXTERNAL_STORAGE

/** Photos frozen when the viewer opens, so re-sorting by a new rating doesn't shift the page under the user. */
private data class ViewerState(val photos: List<Photo>, val start: Int)

@Composable
private fun App(marks: Marks, prefs: SharedPreferences) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var hasPermission by remember {
        mutableStateOf(context.checkSelfPermission(READ_PERMISSION) == PackageManager.PERMISSION_GRANTED)
    }
    val askPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { hasPermission = it }
    LaunchedEffect(Unit) { if (!hasPermission) askPermission.launch(READ_PERMISSION) }
    if (!hasPermission) {
        Box(Modifier.fillMaxSize(), Alignment.Center) {
            Button({ askPermission.launch(READ_PERMISSION) }) { Text("사진 접근 허용") }
        }
        return
    }

    val chosenCover = { bucket: Long -> prefs.getLong("cover_$bucket", -1) }
    var folders by remember { mutableStateOf(loadFolders(context, chosenCover)) }
    var folder by remember { mutableStateOf(folders.firstOrNull { it.id == prefs.getLong("folder", -1) }) }
    var photos by remember { mutableStateOf(emptyList<Photo>()) }
    var reload by remember { mutableIntStateOf(0) }
    LaunchedEffect(folder, reload) {
        folders = withContext(Dispatchers.IO) { loadFolders(context, chosenCover) }
        photos = folder?.let { withContext(Dispatchers.IO) { loadPhotos(context, it.id) } } ?: emptyList()
    }

    var sort by remember(folder) { mutableStateOf(Sort.parse(prefs.getString("sort_${folder?.id}", null))) }

    // AI descriptions: how many photos still lack one, and the outfit groups of those that have one
    val tags = remember { Tags.of(context) }
    val job = Indexer.job?.takeIf { it.bucket == folder?.id }
    var missing by remember(folder) { mutableStateOf<Int?>(null) }
    var groups by remember(folder) { mutableStateOf(emptyList<List<OutfitEntry>>()) }
    var regroup by remember { mutableIntStateOf(0) }
    LaunchedEffect(photos, job?.running, regroup) {
        // also while an analysis runs: a hand-made group must show up right away (descriptions refresh when it ends)
        withContext(Dispatchers.IO) {
            val entries = photos.map {
                val hash = tags.hash(context, it)
                OutfitEntry(it, tags[hash], filenameOutfit(it.name), tags.manualOutfit(hash))
            }
            missing = entries.count { it.tags == null }
            groups = clusterOutfits(entries.filter { it.groupable })
        }
    }
    var naming by remember { mutableStateOf<Pair<Set<Long>, String>?>(null) }
    var renamingOutfit by remember { mutableStateOf<String?>(null) }
    var outfitMode by remember(folder) { mutableStateOf(prefs.getBoolean("outfit_${folder?.id}", false)) }
    var outfit by remember(folder) { mutableStateOf<List<OutfitEntry>?>(null) }
    var confirmIndex by remember { mutableStateOf(false) }
    val askNotify = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {
        folder?.let { Indexer.start(context, it.id) }
    }

    // grouped: the newest photo of each outfit stands for it, with the outfit size as a badge (like the PC)
    val groupOf = remember(groups) {
        groups.associate { group -> group.maxBy { it.photo.date }.photo.id to group.map { it.photo } }
    }
    // the open outfit is a snapshot from when it was opened: drop photos that are gone since (trashed, moved),
    // and fall back to the overview when nothing is left
    val opened = outfit?.let { group ->
        val present = photos.mapTo(HashSet()) { it.id }
        group.filter { it.photo.id in present }.takeIf { it.isNotEmpty() }
    }
    // reads marks, so a rating sort follows rating changes
    val shown = when {
        opened != null -> sortPhotos(opened.map { it.photo }, sort) { marks[it].rating }
        outfitMode -> {
            val grouped = groups.flatMap { g -> g.map { it.photo.id } }.toSet()
            sortPhotos(groupOf.keys.mapNotNull { id -> photos.find { it.id == id } } + photos.filter { it.id !in grouped }, sort) { marks[it].rating }
        }
        else -> sortPhotos(photos, sort) { marks[it].rating }
    }
    /** In grouped view an outfit stands for all its photos, for share, move, trash and questions too. */
    fun expand(ids: Set<Long>) = if (outfitMode && opened == null) ids.flatMap { id -> groupOf[id]?.map { it.id } ?: listOf(id) }.toSet() else ids
    var columns by remember { mutableIntStateOf(prefs.getInt("columns", 4)) }
    var keepRatio by remember { mutableStateOf(prefs.getBoolean("keep_ratio", true)) }
    var selected by remember(folder) { mutableStateOf(emptySet<Long>()) }
    val topGrouped = outfitMode && opened == null
    /** In the grouped overview a tile stands for its whole outfit: tapping it (de)selects every photo of the outfit. */
    fun select(new: Set<Long>) {
        if (!topGrouped) {
            selected = new
            return
        }
        var result = new
        for (id in new - selected) groupOf[id]?.let { result = result + it.map { p -> p.id } }
        for (id in selected - new) groupOf[id]?.let { result = result - it.map { p -> p.id }.toSet() }
        selected = result
    }
    // outfits that have some, but not all, of their photos selected (e.g. picked inside the outfit, then moved up)
    val partial = if (topGrouped) groupOf.filter { (rep, group) -> rep !in selected && group.any { it.id in selected } }.keys else emptySet()
    val gridState = remember(folder) { LazyStaggeredGridState() }
    var viewer by remember { mutableStateOf<ViewerState?>(null) }
    var moving by remember { mutableStateOf(false) }
    // hidden folders: kept out of the folder list until shown again from 숨긴 폴더
    var hidden by remember { mutableStateOf(prefs.getStringSet("hidden_folders", emptySet())!!.mapNotNull(String::toLongOrNull).toSet()) }
    var showHidden by remember { mutableStateOf(false) }
    var renaming by remember { mutableStateOf<Pair<Folder, String>?>(null) }
    var newFolder by remember { mutableStateOf<String?>(null) }
    var pickingCover by remember { mutableStateOf<Folder?>(null) }

    // trash and move go through a system confirmation dialog first
    var afterGrant by remember { mutableStateOf<(() -> Unit)?>(null) }
    val askSystem = rememberLauncherForActivityResult(ActivityResultContracts.StartIntentSenderForResult()) {
        if (it.resultCode == Activity.RESULT_OK) afterGrant?.invoke()
        afterGrant = null
    }
    fun systemRequest(request: PendingIntent, then: () -> Unit) {
        afterGrant = then
        askSystem.launch(IntentSenderRequest.Builder(request.intentSender).build())
    }

    var editing by remember { mutableStateOf<Photo?>(null) }
    val gemini = remember { Gemini(prefs) }
    var settings by remember { mutableStateOf(false) }
    var asking by remember { mutableStateOf<List<Long>?>(null) }

    val current = viewer
    val edited = editing
    when {
        current != null && edited != null && folder != null -> EditorScreen(
            photo = edited,
            folderPath = folder!!.path,
            gemini = gemini,
            onClose = { editing = null },
            onSaved = { saved ->
                // show the new copy right after its original
                val at = current.photos.indexOf(edited) + 1
                viewer = ViewerState(current.photos.toMutableList().apply { add(at, saved) }, at)
                editing = null
                reload++
            },
        )
        current != null -> {
            fun close(index: Int) {
                val photo = current.photos.getOrNull(index)
                val at = shown.indexOf(photo)
                viewer = null
                if (at >= 0 && gridState.layoutInfo.visibleItemsInfo.none { it.index == at }) {
                    scope.launch { gridState.scrollToItem(at) }
                }
            }
            ViewerScreen(
                photos = current.photos,
                start = current.start,
                marks = marks,
                describe = { photo -> tags.cachedHash(photo)?.let { tags[it] }?.let(::tagSummary) },
                onClose = ::close,
                onEdit = { photo ->
                    // come back to the same photo when the editor closes
                    viewer = current.copy(start = current.photos.indexOf(photo))
                    editing = photo
                },
                onShare = { context.startActivity(shareIntent(listOf(it.id))) },
                onTrash = { photo ->
                    systemRequest(trashRequest(context, listOf(photo.id))) {
                        val rest = current.photos - photo
                        viewer = if (rest.isEmpty()) null else current.copy(photos = rest)
                        reload++
                    }
                },
            )
        }
        folder != null -> {
            fun back() = when {
                selected.isNotEmpty() -> selected = emptySet()
                opened != null -> outfit = null
                else -> folder = null
            }
            BackHandler(onBack = ::back)
            GridScreen(
                title = opened?.let { outfitLabel(it.first()) } ?: folder!!.name,
                photos = shown,
                badges = if (outfitMode && opened == null) groupOf.mapValues { it.value.size }.filterValues { it > 1 } else emptyMap(),
                banner = {
                    IndexBanner(job, missing, onStart = {
                        if (gemini.key.isBlank()) Toast.makeText(context, "설정에서 Gemini API 키를 먼저 넣어주세요", Toast.LENGTH_LONG).show()
                        else confirmIndex = true
                    }, onStop = { Indexer.stop(context) })
                },
                marks = marks,
                state = gridState,
                columns = columns,
                onColumns = {
                    columns = it
                    prefs.edit { putInt("columns", it) }
                },
                keepRatio = keepRatio,
                onKeepRatio = {
                    keepRatio = it
                    prefs.edit { putBoolean("keep_ratio", it) }
                },
                sort = sort,
                onSort = {
                    sort = it
                    prefs.edit { putString("sort_${folder?.id}", it.toString()) }
                },
                outfitMode = outfitMode,
                onOutfitMode = {
                    outfitMode = it
                    outfit = null
                    prefs.edit { putBoolean("outfit_${folder?.id}", it) }
                },
                selected = selected,
                onSelect = ::select,
                partial = partial,
                // leave the outfit but keep what is selected, to add photos from outside it
                onUp = if (opened != null && selected.isNotEmpty()) ({ outfit = null }) else null,
                onTitleLongClick = opened?.let { group -> { renamingOutfit = outfitLabel(group.first()) } },
                onOpen = { index ->
                    val group = if (outfitMode && opened == null) groupOf[shown[index].id] else null
                    if (group != null && group.size > 1) outfit = groups.first { g -> g.any { it.photo.id == shown[index].id } }
                    else viewer = ViewerState(shown, index)
                },
                onBack = ::back,
                onShare = { context.startActivity(shareIntent(expand(selected))) },
                onAsk = { asking = expand(selected).toList() },
                onGroup = {
                    val ids = expand(selected)
                    // suggest the name of the outfit most of the selection already belongs to
                    val name = groups.maxByOrNull { g -> g.count { it.photo.id in ids } }
                        ?.takeIf { g -> g.any { it.photo.id in ids } }?.let { outfitLabel(it.first()) }
                        ?.takeIf { it != "기타 의상" } ?: ""
                    naming = ids to name
                },
                onMove = { moving = true },
                onTrash = {
                    systemRequest(trashRequest(context, expand(selected))) {
                        selected = emptySet()
                        reload++
                    }
                },
            )
        }
        else -> FolderScreen(
            folders.filter { (it.id in hidden) == showHidden },
            showHidden = showHidden,
            onShowHidden = { showHidden = it },
            onSettings = { settings = true },
            onRename = { renaming = it to it.name },
            onCover = { pickingCover = it },
            onHide = { f, hide ->
                hidden = if (hide) hidden + f.id else hidden - f.id
                prefs.edit { putStringSet("hidden_folders", hidden.map(Long::toString).toSet()) }
            },
        ) {
            folder = it
            photos = emptyList()
            prefs.edit { putLong("folder", it.id) }
        }
    }

    pickingCover?.let { target ->
        CoverPicker(target, onPick = { photo ->
            prefs.edit { putLong("cover_${target.id}", photo.id) }
            pickingCover = null
            reload++
        }) { pickingCover = null }
    }
    if (settings) SettingsDialog(gemini) { settings = false }
    asking?.let { ids -> AskDialog(gemini, ids) { asking = null } }

    val moveIds = expand(selected)
    fun move(path: String) {
        moving = false
        newFolder = null
        systemRequest(writeRequest(context, moveIds)) {
            scope.launch {
                val failed = withContext(Dispatchers.IO) { moveTo(context, moveIds, path) }
                if (failed > 0) Toast.makeText(context, "${failed}장은 옮기지 못했어요", Toast.LENGTH_SHORT).show()
                selected = emptySet()
                outfit = null
                reload++
            }
        }
    }
    if (moving) {
        AlertDialog(
            onDismissRequest = { moving = false },
            confirmButton = { TextButton({ moving = false }) { Text("취소") } },
            title = { Text("${moveIds.size}장을 옮길 폴더") },
            text = {
                LazyColumn {
                    item {
                        TextButton({
                            moving = false
                            newFolder = ""
                        }) { Text("+ 새 폴더 만들어 옮기기") }
                    }
                    items(folders.filter { it.id != folder?.id }, key = { it.id }) { target ->
                        TextButton({ move(target.path) }) { Text("${target.name}  (${target.count})") }
                    }
                }
            },
        )
    }
    newFolder?.let { name ->
        val path = folder?.let { siblingPath(it, name) }
        AlertDialog(
            onDismissRequest = { newFolder = null },
            confirmButton = { TextButton({ path?.let(::move) }, enabled = path != null) { Text("만들고 옮기기") } },
            dismissButton = { TextButton({ newFolder = null }) { Text("취소") } },
            title = { Text("새 폴더 (${moveIds.size}장)") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(name, { newFolder = it }, label = { Text("폴더 이름") }, singleLine = true)
                    Text(path ?: "폴더 이름에 / \\ : * ? \" < > | 는 쓸 수 없어요", style = MaterialTheme.typography.bodySmall)
                }
            },
        )
    }

    naming?.let { (ids, name) ->
        AlertDialog(
            onDismissRequest = { naming = null },
            confirmButton = {
                TextButton({
                    naming = null
                    scope.launch {
                        withContext(Dispatchers.IO) {
                            tags.setManualOutfit(photos.filter { it.id in ids }.map { tags.hash(context, it) }, name.trim())
                        }
                        selected = emptySet()
                        outfit = null
                        regroup++
                        Toast.makeText(context, "${ids.size}장을 '${name.trim()}'로 묶었어요", Toast.LENGTH_SHORT).show()
                    }
                }, enabled = name.isNotBlank()) { Text("묶기") }
            },
            dismissButton = { TextButton({ naming = null }) { Text("취소") } },
            title = { Text("같은 의상으로 묶기 (${ids.size}장)") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(name, { naming = ids to it }, label = { Text("의상 이름") }, singleLine = true)
                    Text("같은 의상 묶어 보기에서 이 이름으로 한 묶음이 돼요. 파일은 바뀌지 않아요.", style = MaterialTheme.typography.bodySmall)
                }
            },
        )
    }

    renamingOutfit?.let { name ->
        val group = opened
        AlertDialog(
            onDismissRequest = { renamingOutfit = null },
            confirmButton = {
                TextButton({
                    renamingOutfit = null
                    val newName = name.trim()
                    if (group != null) scope.launch {
                        withContext(Dispatchers.IO) { tags.setManualOutfit(group.map { tags.hash(context, it.photo) }, newName) }
                        // the open outfit keeps its photos and shows the new title; it is a hand-made group from now on
                        outfit = group.map { it.copy(manual = newName) }
                        regroup++
                    }
                }, enabled = name.isNotBlank()) { Text("저장") }
            },
            dismissButton = { TextButton({ renamingOutfit = null }) { Text("취소") } },
            title = { Text("의상 이름 변경") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(name, { renamingOutfit = it }, label = { Text("의상 이름") }, singleLine = true)
                    Text("이 묶음의 사진 ${group?.size ?: 0}장이 이 이름의 직접 만든 묶음이 돼요. 나중에 추가되는 사진은 자동으로 들어오지 않아요.", style = MaterialTheme.typography.bodySmall)
                }
            },
        )
    }

    renaming?.let { (target, name) ->
        val path = siblingPath(target, name)
        val changed = name.trim() != target.name
        AlertDialog(
            onDismissRequest = { renaming = null },
            confirmButton = {
                TextButton({
                    renaming = null
                    val ids = loadPhotos(context, target.id).map { it.id }
                    systemRequest(writeRequest(context, ids)) {
                        scope.launch {
                            val failed = withContext(Dispatchers.IO) { renameFolder(context, target, name) }
                            if (failed > 0) Toast.makeText(context, "${failed}장은 옮기지 못했어요", Toast.LENGTH_SHORT).show()
                            // the folder id comes from its path, so carry the hidden mark over to the new id
                            val moved = withContext(Dispatchers.IO) { loadFolders(context) }.firstOrNull { it.path == path }
                            if (target.id in hidden && moved != null) {
                                hidden = hidden - target.id + moved.id
                                prefs.edit { putStringSet("hidden_folders", hidden.map(Long::toString).toSet()) }
                            }
                            reload++
                        }
                    }
                }, enabled = path != null && changed) { Text("변경") }
            },
            dismissButton = { TextButton({ renaming = null }) { Text("취소") } },
            title = { Text("폴더 이름 변경") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(name, { renaming = target to it }, label = { Text("새 이름") }, singleLine = true)
                    Text(
                        path?.let { "${target.path} → $it\n사진 ${target.count}장을 옮겨요. 사진이 아닌 파일은 원래 폴더에 남아요." }
                            ?: "폴더 이름에 / \\ : * ? \" < > | 는 쓸 수 없어요",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            },
        )
    }

    if (confirmIndex) {
        AlertDialog(
            onDismissRequest = { confirmIndex = false },
            confirmButton = {
                TextButton({
                    confirmIndex = false
                    // progress shows in a notification while the screen is off; works without it too
                    if (Build.VERSION.SDK_INT >= 33) askNotify.launch(Manifest.permission.POST_NOTIFICATIONS)
                    else folder?.let { Indexer.start(context, it.id) }
                }) { Text("분석") }
            },
            dismissButton = { TextButton({ confirmIndex = false }) { Text("취소") } },
            title = { Text("AI 분석 ${missing ?: 0}장") },
            text = {
                Text("분석 안 된 사진만 8장씩 묶어 Gemini로 보내 의상·헤어·자세를 기록해요. " +
                    "화면을 꺼도 계속되고, 무료 한도에 걸리면 멈췄다가 다음에 누르면 이어서 해요.")
            },
        )
    }
}

@Composable
private fun IndexBanner(job: IndexJob?, missing: Int?, onStart: () -> Unit, onStop: () -> Unit) {
    val text = when {
        job?.running == true -> if (job.total == 0) "AI 분석 준비 중…" else "AI 분석 중 ${job.done} / ${job.total}"
        missing == null -> "AI 분석 상태 확인 중…"
        missing > 0 -> "AI 분석 안 된 사진 ${missing}장"
        else -> return
    }
    Column(Modifier.fillMaxWidth().padding(horizontal = 16.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(text, Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
            if (job?.running == true) TextButton(onStop) { Text("중지") }
            else if ((missing ?: 0) > 0) TextButton(onStart) { Text("분석") }
        }
        if (job?.running == true && job.total > 0) LinearProgressIndicator({ job.done / job.total.toFloat() }, Modifier.fillMaxWidth())
        if (job?.running == false && job.error.isNotEmpty()) {
            Text(job.error, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable
private fun SettingsDialog(gemini: Gemini, onDismiss: () -> Unit) {
    var key by remember { mutableStateOf(gemini.key) }
    var model by remember { mutableStateOf(gemini.model) }
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton({
                gemini.key = key
                gemini.model = model
                onDismiss()
            }) { Text("저장") }
        },
        dismissButton = { TextButton(onDismiss) { Text("취소") } },
        title = { Text("Gemini 설정") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(
                    key, { key = it }, label = { Text("API 키 (aistudio.google.com)") }, singleLine = true,
                    visualTransformation = PasswordVisualTransformation(),
                )
                OutlinedTextField(model, { model = it }, label = { Text("모델") }, singleLine = true)
                Text(
                    "편집 명령은 글자만 보내요. 사진은 `AI 자동` 보정과 `질문`을 직접 눌렀을 때만 그 사진을 보내요. " +
                        "무료 tier에서는 보낸 내용을 Google이 서비스 개선에 쓸 수 있어요.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        },
    )
}

@Composable
private fun AskDialog(gemini: Gemini, ids: List<Long>, onDismiss: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var question by remember { mutableStateOf("") }
    var answer by remember { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }
    AlertDialog(
        onDismissRequest = { if (!busy) onDismiss() },
        confirmButton = {
            TextButton({
                busy = true
                scope.launch {
                    answer = try {
                        withContext(Dispatchers.IO) { gemini.ask(context, ids.map(::imageUri), question.trim()) }
                    } catch (e: GeminiError) {
                        e.message!!
                    } catch (e: java.io.IOException) {
                        "Gemini에 연결하지 못했어요: ${e.message}"
                    }
                    busy = false
                }
            }, enabled = question.isNotBlank() && !busy) { Text("보내기") }
        },
        dismissButton = { TextButton(onDismiss, enabled = !busy) { Text("닫기") } },
        title = { Text("사진 ${ids.size}장에 질문") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("선택한 순서대로 첫번째, 두번째…로 보내요.", style = MaterialTheme.typography.bodySmall)
                OutlinedTextField(question, { question = it }, placeholder = { Text("예: 두 사람이 얼마나 비슷해?") }, enabled = !busy)
                if (busy) CircularProgressIndicator()
                if (answer.isNotEmpty()) SelectionContainer { Text(answer) }
            }
        },
    )
}
