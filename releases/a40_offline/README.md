# A40 离线安装包（方式 B）

x86 + NVIDIA A40 离线性能对比用压缩包，用法见仓库根目录 `scripts/README_A40_OFFLINE.md`。

| 文件 | 说明 |
|------|------|
| `pointcept_a40_offline_<commit>_<date>.tar.gz` | 离线包（含 pointops / pointops2 / pointops_torch 源码与 install.sh） |
| `*.tar.gz.sha256` | 校验和 |

```bash
sha256sum -c pointcept_a40_offline_*.tar.gz.sha256
tar xzf pointcept_a40_offline_*.tar.gz
cd pointcept_a40_offline_* && bash install.sh
```
