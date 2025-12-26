# EmoVoice-Plus Demo

> 目录结构（已按你的截图对齐）：
> - `README.md` 在仓库根目录
> - 音频文件都在 `demo/` 目录下：`demo/1-1.wav ~ demo/4-2.wav`
>
> GitHub 网页里要让 `<audio>` 正常播放，必须用 `?raw=1` 拿到原始音频流（否则会读取到 blob 预览页 HTML，显示 0:00）。

---

## 注入 neutral → happy 激活方向值

<table style="width:100%; table-layout:fixed;">
  <colgroup>
    <col style="width:70%;">
    <col style="width:15%;">
    <col style="width:15%;">
  </colgroup>

  <tr>
    <th align="left">内容</th>
    <th align="center">左侧（*-1）</th>
    <th align="center">右侧（*-2）</th>
  </tr>

  <tr>
    <td style="word-break:break-word;">
      <b>文本：</b> A day spent laughing with friends is a day well spent.<br/>
      <b>情绪描述：</b> Communal joy found within the comfort of familiar, happy faces.
    </td>
    <td align="center">
      <audio controls preload="none" src="demo/1-1.wav?raw=1" style="width:100%; max-width:220px;"></audio><br/>
      <a href="demo/1-1.wav?raw=1">▶️ demo/1-1.wav</a>
    </td>
    <td align="center">
      <audio controls preload="none" src="demo/1-2.wav?raw=1" style="width:100%; max-width:220px;"></audio><br/>
      <a href="demo/1-2.wav?raw=1">▶️ demo/1-2.wav</a>
    </td>
  </tr>

  <tr>
    <td style="word-break:break-word;">
      <b>文本：</b> Look at that rainbow! It's breathtaking and perfectly magical.<br/>
      <b>情绪描述：</b> A spontaneous, shared moment of wonder and joy at nature's beauty.
    </td>
    <td align="center">
      <audio controls preload="none" src="demo/2-1.wav?raw=1" style="width:100%; max-width:220px;"></audio><br/>
      <a href="demo/2-1.wav?raw=1">▶️ demo/2-1.wav</a>
    </td>
    <td align="center">
      <audio controls preload="none" src="demo/2-2.wav?raw=1" style="width:100%; max-width:220px;"></audio><br/>
      <a href="demo/2-2.wav?raw=1">▶️ demo/2-2.wav</a>
    </td>
  </tr>

  <tr>
    <td style="word-break:break-word;">
      <b>文本：</b> She spun around in the meadow, arms outstretched to embrace the sunlit sky.<br/>
      <b>情绪描述：</b> Articulating an unbridled joy that embraces the world with spontaneous, exuberant motion.
    </td>
    <td align="center">
      <audio controls preload="none" src="demo/3-1.wav?raw=1" style="width:100%; max-width:220px;"></audio><br/>
      <a href="demo/3-1.wav?raw=1">▶️ demo/3-1.wav</a>
    </td>
    <td align="center">
      <audio controls preload="none" src="demo/3-2.wav?raw=1" style="width:100%; max-width:220px;"></audio><br/>
      <a href="demo/3-2.wav?raw=1">▶️ demo/3-2.wav</a>
    </td>
  </tr>
</table>

---

## 注入 happy → sad 激活方向值

<table style="width:100%; table-layout:fixed;">
  <colgroup>
    <col style="width:70%;">
    <col style="width:15%;">
    <col style="width:15%;">
  </colgroup>

  <tr>
    <th align="left">内容</th>
    <th align="center">左侧（*-1）</th>
    <th align="center">右侧（*-2）</th>
  </tr>

  <tr>
    <td style="word-break:break-word;">
      <b>文本：</b> She spun around in the meadow, arms outstretched to embrace the sunlit sky.<br/>
      <b>情绪描述：</b> Articulating an unbridled joy that embraces the world with spontaneous, exuberant motion.
    </td>
    <td align="center">
      <audio controls preload="none" src="demo/4-1.wav?raw=1" style="width:100%; max-width:220px;"></audio><br/>
      <a href="demo/4-1.wav?raw=1">▶️ demo/4-1.wav</a>
    </td>
    <td align="center">
      <audio controls preload="none" src="demo/4-2.wav?raw=1" style="width:100%; max-width:220px;"></audio><br/>
      <a href="demo/4-2.wav?raw=1">▶️ demo/4-2.wav</a>
    </td>
  </tr>
</table>
