# EmoVoice-Plus Demo

> 目录结构：本 README 位于 `demo/` 目录下，并与 `1-1.wav ~ 4-2.wav` 同级。
>
> 你在 GitHub 网页里点播放/链接会“跳转到网页”，原因是 GitHub 默认打开的是 `blob` 预览页（HTML），不是音频原始数据。
> 下面的写法为每个音频提供 **两种 source**：先尝试本地相对路径（适用于本地预览/静态站点），失败则回退到 `?raw=1`（适用于 GitHub 网页）。

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
      <audio controls preload="none" style="width:100%; max-width:220px;">
        <source src="1-1.wav" type="audio/wav"/>
        <source src="1-1.wav?raw=1" type="audio/wav"/>
      </audio><br/>
      <a href="1-1.wav">▶️ 1-1.wav</a> · <a href="1-1.wav?raw=1">raw</a>
    </td>
    <td align="center">
      <audio controls preload="none" style="width:100%; max-width:220px;">
        <source src="1-2.wav" type="audio/wav"/>
        <source src="1-2.wav?raw=1" type="audio/wav"/>
      </audio><br/>
      <a href="1-2.wav">▶️ 1-2.wav</a> · <a href="1-2.wav?raw=1">raw</a>
    </td>
  </tr>

  <tr>
    <td style="word-break:break-word;">
      <b>文本：</b> Look at that rainbow! It's breathtaking and perfectly magical.<br/>
      <b>情绪描述：</b> A spontaneous, shared moment of wonder and joy at nature's beauty.
    </td>
    <td align="center">
      <audio controls preload="none" style="width:100%; max-width:220px;">
        <source src="2-1.wav" type="audio/wav"/>
        <source src="2-1.wav?raw=1" type="audio/wav"/>
      </audio><br/>
      <a href="2-1.wav">▶️ 2-1.wav</a> · <a href="2-1.wav?raw=1">raw</a>
    </td>
    <td align="center">
      <audio controls preload="none" style="width:100%; max-width:220px;">
        <source src="2-2.wav" type="audio/wav"/>
        <source src="2-2.wav?raw=1" type="audio/wav"/>
      </audio><br/>
      <a href="2-2.wav">▶️ 2-2.wav</a> · <a href="2-2.wav?raw=1">raw</a>
    </td>
  </tr>

  <tr>
    <td style="word-break:break-word;">
      <b>文本：</b> She spun around in the meadow, arms outstretched to embrace the sunlit sky.<br/>
      <b>情绪描述：</b> Articulating an unbridled joy that embraces the world with spontaneous, exuberant motion.
    </td>
    <td align="center">
      <audio controls preload="none" style="width:100%; max-width:220px;">
        <source src="3-1.wav" type="audio/wav"/>
        <source src="3-1.wav?raw=1" type="audio/wav"/>
      </audio><br/>
      <a href="3-1.wav">▶️ 3-1.wav</a> · <a href="3-1.wav?raw=1">raw</a>
    </td>
    <td align="center">
      <audio controls preload="none" style="width:100%; max-width:220px;">
        <source src="3-2.wav" type="audio/wav"/>
        <source src="3-2.wav?raw=1" type="audio/wav"/>
      </audio><br/>
      <a href="3-2.wav">▶️ 3-2.wav</a> · <a href="3-2.wav?raw=1">raw</a>
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
      <audio controls preload="none" style="width:100%; max-width:220px;">
        <source src="4-1.wav" type="audio/wav"/>
        <source src="4-1.wav?raw=1" type="audio/wav"/>
      </audio><br/>
      <a href="4-1.wav">▶️ 4-1.wav</a> · <a href="4-1.wav?raw=1">raw</a>
    </td>
    <td align="center">
      <audio controls preload="none" style="width:100%; max-width:220px;">
        <source src="4-2.wav" type="audio/wav"/>
        <source src="4-2.wav?raw=1" type="audio/wav"/>
      </audio><br/>
      <a href="4-2.wav">▶️ 4-2.wav</a> · <a href="4-2.wav?raw=1">raw</a>
    </td>
  </tr>
</table>
