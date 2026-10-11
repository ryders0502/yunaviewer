package io.github.ryders0502.yunaviewer

import android.graphics.Bitmap
import android.widget.Toast
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.OutlinedTextField
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.ui.res.painterResource
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.foundation.systemGestureExclusion
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.abs
import kotlin.math.roundToInt

private enum class Tool(val label: String, val icon: Int) {
    CROP("자르기", R.drawable.ic_crop), ADJUST("조정", R.drawable.ic_tune), BAND("구간 확장", R.drawable.ic_height)
}

private const val PREVIEW_PIXELS = 2_000_000
private val Accent = Color(0xFFFFD54D)
private val BandColor = Color(0xFFFF6B8B)
private val Scrim = Color(0xB3000000)

/** Preview bitmap and whether it is the uncropped photo the crop box and band lines are drawn over. */
private class Preview(val bitmap: Bitmap, val uncropped: Boolean)

@Composable
fun EditorScreen(photo: Photo, folderPath: String, gemini: Gemini, onClose: () -> Unit, onSaved: (Photo) -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var source by remember { mutableStateOf<Bitmap?>(null) }
    LaunchedEffect(photo.id) {
        source = runCatching { withContext(Dispatchers.IO) { decode(context, photo.uri, PREVIEW_PIXELS) } }.getOrElse {
            Toast.makeText(context, "사진을 열지 못했어요", Toast.LENGTH_SHORT).show()
            onClose()
            return@LaunchedEffect
        }
    }

    var params by remember { mutableStateOf(EditParams()) }
    var history by remember { mutableStateOf(listOf(EditParams())) }
    var at by remember { mutableIntStateOf(0) }
    fun commit(p: EditParams = params) {
        params = p
        if (p != history[at]) {
            history = history.take(at + 1) + p
            at = history.lastIndex
        }
    }

    var tool by remember { mutableStateOf(Tool.ADJUST) }
    var control by remember { mutableStateOf(CONTROLS[0]) }
    var showOriginal by remember { mutableStateOf(false) }
    var saving by remember { mutableStateOf(false) }
    var confirmClose by remember { mutableStateOf(false) }
    val dirty = params != EditParams()
    val uncropped = tool != Tool.ADJUST
    var preview by remember { mutableStateOf<Preview?>(null) }
    LaunchedEffect(source, params, uncropped) {
        val src = source ?: return@LaunchedEffect
        preview = Preview(withContext(Dispatchers.Default) { render(src, params, crop = !uncropped) }, uncropped)
    }

    fun close() = if (dirty) confirmClose = true else onClose()
    fun save() {
        saving = true
        scope.launch {
            runCatching { withContext(Dispatchers.Default) { saveEdited(context, photo, folderPath, params) } }
                .onSuccess {
                    Toast.makeText(context, "${it.name} 저장", Toast.LENGTH_SHORT).show()
                    onSaved(it)
                }
                .onFailure {
                    saving = false
                    Toast.makeText(context, "저장 실패: ${it.message ?: it.javaClass.simpleName}", Toast.LENGTH_LONG).show()
                }
        }
    }
    BackHandler(enabled = !saving) { close() }

    var aiBusy by remember { mutableStateOf(false) }
    var command by remember { mutableStateOf("") }
    fun toast(text: String) = Toast.makeText(context, text, Toast.LENGTH_LONG).show()
    fun runAi(block: suspend () -> Unit) {
        aiBusy = true
        scope.launch {
            try {
                block()
            } catch (e: GeminiError) {
                toast(e.message!!)
            } catch (e: java.io.IOException) {
                toast("Gemini에 연결하지 못했어요: ${e.message}")
            } finally {
                aiBusy = false
            }
        }
    }
    // sends the photo as currently cropped, with neutral colors, like the PC
    suspend fun enhance() {
        val src = source ?: return
        val base = params
        val (values, message) = withContext(Dispatchers.IO) { gemini.enhance(render(src, base.neutralColors())) }
        commit(params.withColors(values))
        if (message.isNotEmpty()) toast(message)
    }
    fun sendCommand() {
        val text = command.trim()
        if (text.isEmpty()) return
        runAi {
            val src = source ?: return@runAi
            val result = withContext(Dispatchers.IO) { gemini.editCommand(text, params) }
            commit(applyCommand(params, result, src.width, src.height))
            command = ""
            if (result.message.isNotEmpty()) toast(result.message)
            if (result.autoEnhance) enhance()
        }
    }
    val busy = saving || aiBusy

    // view zoom: two fingers zoom and move the photo (out to half size, so the panels can be cleared);
    // one finger drags the crop box or band lines, or moves a zoomed photo while adjusting. Double tap resets.
    var zoom by remember { mutableFloatStateOf(1f) }
    var pan by remember { mutableStateOf(Offset.Zero) }
    var chrome by remember { mutableStateOf(true) }
    val currentTool by rememberUpdatedState(tool)

    CompositionLocalProvider(LocalContentColor provides Color.White) {
        Box(Modifier.fillMaxSize().background(Color.Black), contentAlignment = Alignment.Center) {
            val shown = preview
            val original = source
            // panels sit above this layer, so their taps and slider drags never reach these gestures
            Box(
                Modifier
                    .fillMaxSize()
                    .pointerInput(Unit) {
                        awaitEachGesture {
                            awaitFirstDown(requireUnconsumed = false)
                            do {
                                val event = awaitPointerEvent(PointerEventPass.Initial)
                                val fingers = event.changes.count { it.pressed }
                                if (fingers >= 2 || (zoom > 1f && currentTool == Tool.ADJUST)) {
                                    val z = event.calculateZoom()
                                    val move = event.calculatePan()
                                    if (z != 1f || move != Offset.Zero) {
                                        zoom = (zoom * z).coerceIn(0.5f, 5f)
                                        val limit = Offset(size.width * zoom / 2, size.height * zoom / 2)
                                        pan = (pan + move).let { Offset(it.x.coerceIn(-limit.x, limit.x), it.y.coerceIn(-limit.y, limit.y)) }
                                        event.changes.forEach { it.consume() }
                                    }
                                }
                            } while (event.changes.any { it.pressed })
                        }
                    }
                    .pointerInput(Unit) {
                        detectTapGestures(
                            onTap = { chrome = !chrome },
                            onDoubleTap = {
                                zoom = 1f
                                pan = Offset.Zero
                            },
                        )
                    }
                    .graphicsLayer {
                    scaleX = zoom
                    scaleY = zoom
                    translationX = pan.x
                    translationY = pan.y
                },
                contentAlignment = Alignment.Center,
            ) {
                if (showOriginal && original != null) {
                    Image(original.asImageBitmap(), null, Modifier.aspectRatio(original.width / original.height.toFloat()))
                } else if (shown != null) {
                    // overlays sit inside the zoomed layer, so their drags land in photo coordinates at any zoom
                    Box(Modifier.aspectRatio(shown.bitmap.width / shown.bitmap.height.toFloat())) {
                        Image(shown.bitmap.asImageBitmap(), null, Modifier.fillMaxSize())
                        if (shown.uncropped && !busy) when (tool) {
                            Tool.CROP -> CropOverlay(params, onChange = { params = it }, onDone = { commit() })
                            Tool.BAND -> BandOverlay(params, onChange = { params = it }, onDone = { commit() })
                            Tool.ADJUST -> {}
                        }
                    }
                }
            }
            if (shown == null || busy) CircularProgressIndicator()

            if (chrome) {
                Row(
                    Modifier.align(Alignment.TopCenter).fillMaxWidth().background(Scrim).statusBarsPadding(),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    IconButton(::close, enabled = !saving) { Icon(Icons.Filled.Close, "닫기") }
                    Spacer(Modifier.weight(1f))
                    TextButton({ params = history[--at] }, enabled = at > 0 && !busy) { Text("↶ 취소") }
                    TextButton({ params = history[++at] }, enabled = at < history.lastIndex && !busy) { Text("↷ 다시") }
                    // hold to compare with the untouched photo
                    Text(
                        "원본",
                        Modifier
                            .pointerInput(Unit) {
                                detectTapGestures(onPress = {
                                    showOriginal = true
                                    tryAwaitRelease()
                                    showOriginal = false
                                })
                            }
                            .padding(12.dp),
                    )
                    TextButton(::save, enabled = dirty && !busy) { Text("저장", color = if (dirty && !busy) Accent else Color.Gray) }
                }

                Column(
                    Modifier.align(Alignment.BottomCenter).fillMaxWidth().background(Scrim)
                        .navigationBarsPadding().imePadding(),
                ) {
                    when (tool) {
                        Tool.CROP -> Row(Modifier.fillMaxWidth().padding(horizontal = 8.dp)) {
                            TextButton({
                                // the crop box is in the old orientation's coordinates, so start over
                                commit(params.copy(rot = (params.rot + 90) % 360, cropL = 0f, cropT = 0f, cropR = 1f, cropB = 1f))
                            }) { Text("↻ 회전") }
                            TextButton({ commit(params.copy(flip = !params.flip, cropL = 1 - params.cropR, cropR = 1 - params.cropL)) }) { Text("⇋ 반전") }
                            Spacer(Modifier.weight(1f))
                            TextButton({ commit(params.copy(cropL = 0f, cropT = 0f, cropR = 1f, cropB = 1f)) }, enabled = params.cropped) { Text("자르기 해제") }
                        }
                        Tool.ADJUST -> Column {
                            Row(Modifier.fillMaxWidth()) {
                                IconButton({ runAi { enhance() } }, Modifier.weight(1f), enabled = !busy) {
                                    Icon(painterResource(R.drawable.ic_auto_fix_high), "AI 자동 보정", tint = BandColor)
                                }
                                for (c in CONTROLS) {
                                    // a dot under the icon marks a control that is not at its neutral value
                                    IconButton({ control = c }, Modifier.weight(1f)) {
                                        Box(contentAlignment = Alignment.BottomCenter) {
                                            Icon(painterResource(c.icon), c.label, tint = if (c == control) Accent else Color.White)
                                            if (c.get(params) != c.neutral) Box(Modifier.offset(y = 8.dp).size(4.dp).background(Accent, CircleShape))
                                        }
                                    }
                                }
                            }
                            ValueSlider(
                                value = control.get(params),
                                range = control.range,
                                text = control.label + " " + control.text(params),
                                onChange = { params = control.set(params, it) },
                                onDone = { commit() },
                                onReset = { commit(control.set(params, control.neutral)) },
                            )
                        }
                        Tool.BAND -> Column {
                            Text(
                                "분홍 선 두 개로 구간을 정하면 그 사이만 배경째 세로로 늘거나 줄어요. " +
                                    "다리는 엉덩이 바로 아래~발목, 상체는 어깨 아래~허리에 맞춰보세요.",
                                Modifier.padding(horizontal = 16.dp),
                                style = MaterialTheme.typography.bodySmall,
                            )
                            ValueSlider(
                                value = params.stretch,
                                range = -MAX_STRETCH_PCT..MAX_STRETCH_PCT,
                                text = "%+d%%".format(params.stretch.roundToInt()),
                                onChange = { params = params.copy(stretch = it.roundToInt().toFloat()) },
                                onDone = { commit() },
                                onReset = { commit(params.copy(stretch = 0f)) },
                            )
                        }
                    }

                    // text only goes out here; AI 자동 is the one that sends the photo
                    OutlinedTextField(
                        command, { command = it }, Modifier.fillMaxWidth().padding(horizontal = 8.dp),
                        placeholder = { Text("예: 조금 밝고 따뜻하게, 4:5로 잘라줘") },
                        trailingIcon = {
                            IconButton(::sendCommand, enabled = command.isNotBlank() && !busy) {
                                Icon(Icons.AutoMirrored.Filled.Send, "실행")
                            }
                        },
                        singleLine = true,
                        enabled = !busy,
                        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                        keyboardActions = KeyboardActions(onSend = { sendCommand() }),
                    )

                    Row(Modifier.fillMaxWidth()) {
                        Tool.entries.forEach { t ->
                            IconButton({ tool = t }, Modifier.weight(1f)) {
                                Icon(painterResource(t.icon), t.label, tint = if (t == tool) Accent else Color.White.copy(alpha = 0.6f))
                            }
                        }
                    }
                }
            }
        }
    }

    if (confirmClose) {
        AlertDialog(
            onDismissRequest = { confirmClose = false },
            confirmButton = { TextButton(onClose) { Text("버리기") } },
            dismissButton = { TextButton({ confirmClose = false }) { Text("계속 편집") } },
            text = { Text("저장하지 않은 편집을 버릴까요?") },
        )
    }
}

/** Full-width slider with its value above; tapping the value resets it. */
@Composable
private fun ValueSlider(
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    text: String,
    onChange: (Float) -> Unit,
    onDone: () -> Unit,
    onReset: () -> Unit,
) {
    Column(Modifier.fillMaxWidth().padding(horizontal = 8.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        Text(text, Modifier.clickable(onClick = onReset).padding(horizontal = 12.dp), style = MaterialTheme.typography.bodySmall)
        Slider(value, onChange, Modifier.fillMaxWidth(), valueRange = range, onValueChangeFinished = onDone)
    }
}

@Composable
private fun CropOverlay(p: EditParams, onChange: (EditParams) -> Unit, onDone: () -> Unit) {
    val current by rememberUpdatedState(p)
    val change by rememberUpdatedState(onChange)
    val done by rememberUpdatedState(onDone)
    val grab = with(LocalDensity.current) { 36.dp.toPx() }
    Canvas(
        Modifier
            .fillMaxSize()
            // the photo fills the screen width, so corners sit on the back-gesture edges: keep them for the crop
            .then(listOf(0, 1, 2, 3).fold(Modifier as Modifier) { m, corner ->
                m.systemGestureExclusion { box ->
                    val x = if (corner == 1 || corner == 2) p.cropR else p.cropL
                    val y = if (corner >= 2) p.cropB else p.cropT
                    Rect(Offset(x * box.size.width, y * box.size.height), grab)
                }
            })
            .pointerInput(Unit) {
                var handle = -1
                detectDragGestures(
                    onDragStart = { at ->
                        val c = current
                        val (w, h) = size.width.toFloat() to size.height.toFloat()
                        val corners = listOf(Offset(c.cropL * w, c.cropT * h), Offset(c.cropR * w, c.cropT * h),
                                             Offset(c.cropR * w, c.cropB * h), Offset(c.cropL * w, c.cropB * h))
                        val inside = at.x in c.cropL * w..c.cropR * w && at.y in c.cropT * h..c.cropB * h
                        handle = corners.indexOfFirst { (it - at).getDistance() < grab }.takeIf { it >= 0 } ?: if (inside) 4 else -1
                    },
                    onDragEnd = { done() },
                    onDragCancel = { done() },
                    onDrag = { pointer, drag ->
                        pointer.consume()
                        change(moveCrop(current, handle, drag.x / size.width, drag.y / size.height))
                    },
                )
            },
    ) {
        val l = p.cropL * size.width
        val t = p.cropT * size.height
        val r = p.cropR * size.width
        val b = p.cropB * size.height
        val dim = Color(0x99000000)
        drawRect(dim, Offset.Zero, Size(size.width, t))
        drawRect(dim, Offset(0f, b), Size(size.width, size.height - b))
        drawRect(dim, Offset(0f, t), Size(l, b - t))
        drawRect(dim, Offset(r, t), Size(size.width - r, b - t))
        drawRect(Color.White, Offset(l, t), Size(r - l, b - t), style = Stroke(2.dp.toPx()))
        for (corner in listOf(Offset(l, t), Offset(r, t), Offset(r, b), Offset(l, b))) drawCircle(Color.White, 7.dp.toPx(), corner)
    }
}

@Composable
private fun BandOverlay(p: EditParams, onChange: (EditParams) -> Unit, onDone: () -> Unit) {
    val current by rememberUpdatedState(p)
    val change by rememberUpdatedState(onChange)
    val done by rememberUpdatedState(onDone)
    val grab = with(LocalDensity.current) { 40.dp.toPx() }
    Canvas(
        Modifier
            .fillMaxSize()
            .then(listOf(true, false).fold(Modifier as Modifier) { m, top ->
                m.systemGestureExclusion { box ->
                    val (t, b) = bandLines(p)
                    Rect(Offset(box.size.width - grab, (if (top) t else b) * box.size.height), grab)
                }
            })
            .pointerInput(Unit) {
                var top: Boolean? = null
                detectDragGestures(
                    onDragStart = { at ->
                        val (t, b) = bandLines(current)
                        val dt = abs(at.y - t * size.height)
                        val db = abs(at.y - b * size.height)
                        top = if (minOf(dt, db) > grab) null else dt < db
                    },
                    onDragEnd = { done() },
                    onDragCancel = { done() },
                    onDrag = { pointer, _ ->
                        val which = top ?: return@detectDragGestures
                        pointer.consume()
                        change(moveBandLine(current, which, (pointer.position.y / size.height).coerceIn(0f, 1f)))
                    },
                )
            },
    ) {
        val (t, b) = bandLines(p)
        val y1 = t * size.height
        val y2 = b * size.height
        drawRect(BandColor.copy(alpha = 0.18f), Offset(0f, y1), Size(size.width, y2 - y1))
        for (y in listOf(y1, y2)) {
            drawLine(BandColor, Offset(0f, y), Offset(size.width, y), 2.dp.toPx())
            drawCircle(BandColor, 9.dp.toPx(), Offset(size.width - 16.dp.toPx(), y))
        }
    }
}
