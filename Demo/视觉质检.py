"""环形端子传送带质检的业务流程演示。

这不是检测模型。函数体是业务步骤，输入用模拟帧代替相机。
对应《技术拆解/01-视觉质检.md》的交付示例。

主流程：

    光学冻结？
      否 → 停止收数，先改光
      是 → 每一帧：
            采集（曝光已锁） → 检测框 → 跟踪成「一件」
            → 判定放行 / 剔除 / 复判
            → 影子模式只显示；自动模式才写 PLC
            → 原图和判定落库
    心跳丢失 → 停带。不允许模型缺失时全部放行。
    复判改过的标签当晚回训练机，工位不在线更新模型。
"""

from dataclasses import dataclass, field
from enum import Enum


class Label(Enum):
    GOOD = "合格"
    HALF_RING = "半环"
    NO_PLASTIC = "无塑料"
    NO_RING = "无环"


class Decision(Enum):
    PASS = "放行"
    REJECT = "剔除"
    REVIEW = "复判"


class Phase(Enum):
    SHADOW = "影子模式"  # 只画框，气缸不接
    AUTO = "有限自动"  # 高置信度才剔除


# 漏检比误杀贵，但低置信度仍不自动剔除。数字是示例门槛，不是实测。
REJECT_CONFIDENCE = 0.85


@dataclass
class Box:
    label: Label
    confidence: float
    track_id: int


@dataclass
class Frame:
    """一帧相机。同一件零件会连续出现在多帧里。"""

    index: int
    boxes: list[Box]
    camera_ok: bool = True


@dataclass
class PartRecord:
    track_id: int
    label: Label
    confidence: float
    decision: Decision
    plc_acted: bool
    note: str


@dataclass
class Station:
    phase: Phase
    optical_frozen: bool
    heartbeat_ok: bool = True
    records: list[PartRecord] = field(default_factory=list)
    review_queue: list[PartRecord] = field(default_factory=list)
    seen_tracks: set[int] = field(default_factory=set)


def assert_optical_frozen(station: Station) -> None:
    """光源、镜头、高度、曝光冻结之后才允许收数和判定。"""
    if not station.optical_frozen:
        raise RuntimeError("光学未冻结：先改光，不改网络，也不要把旧标注拿来训练。")
    print("1. 光学冻结单有效，曝光与光源锁定。")


def acquire(frame: Frame) -> Frame:
    """相机只看传送带的一段。触发、曝光、增益在工位上是固定的。"""
    print(f"2. 采集第 {frame.index} 帧。")
    if not frame.camera_ok:
        raise RuntimeError("相机无图：停带，禁止全部放行。")
    return frame


def detect(frame: Frame) -> list[Box]:
    """边缘上的小检测器。这里直接返回模拟框，代表已编译模型的输出。"""
    print(f"3. 检测：本帧 {len(frame.boxes)} 个框。")
    return frame.boxes


def track(station: Station, boxes: list[Box]) -> list[Box]:
    """跨帧把同一件收成一个 track。已经判定过的 track 不再计数。"""
    fresh = []
    for box in boxes:
        if box.track_id in station.seen_tracks:
            print(f"4. 跟踪：件 {box.track_id} 仍在视野里，不重复计数。")
            continue
        fresh.append(box)
    return fresh


def judge(box: Box) -> Decision:
    """合格放行；缺陷要够置信才剔除；其余进复判，不进拨杆。"""
    if box.label is Label.GOOD and box.confidence >= REJECT_CONFIDENCE:
        return Decision.PASS
    if box.label is not Label.GOOD and box.confidence >= REJECT_CONFIDENCE:
        return Decision.REJECT
    return Decision.REVIEW


def write_plc(station: Station, decision: Decision) -> bool:
    """PLC 只收放行 / 剔除 / 复判 / 心跳。影子模式不驱动气缸。"""
    if not station.heartbeat_ok:
        print("5. 心跳丢失：停带。不默认放行。")
        return False
    if station.phase is Phase.SHADOW:
        print(f"5. {station.phase.value}：屏幕显示「{decision.value}」，PLC 不动作。")
        return False
    print(f"5. {station.phase.value}：向 PLC 发送「{decision.value}」。")
    return True


def save_local(record: PartRecord) -> None:
    """原图、框、类别、置信度、批次留在工位，供复判和按周重训。"""
    extra = f" {record.note}" if record.note else ""
    print(
        f"6. 落库：件 {record.track_id} {record.label.value} "
        f"置信度 {record.confidence:.2f} → {record.decision.value}。{extra}"
    )


def inspect_frame(station: Station, frame: Frame) -> None:
    image = acquire(frame)
    boxes = detect(image)
    for box in track(station, boxes):
        decision = judge(box)
        acted = write_plc(station, decision)
        if not station.heartbeat_ok:
            decision = Decision.REVIEW
            note = "心跳丢失，停带，不按模型放行"
        else:
            note = ""
        record = PartRecord(
            track_id=box.track_id,
            label=box.label,
            confidence=box.confidence,
            decision=decision,
            plc_acted=acted,
            note=note,
        )
        station.seen_tracks.add(box.track_id)
        station.records.append(record)
        if decision is Decision.REVIEW:
            station.review_queue.append(record)
        save_local(record)


def review_and_queue_retrain(station: Station) -> None:
    """人在复判台改标签。工位不在线学习，增量晚上回训练机。"""
    if not station.review_queue:
        print("7. 本段没有复判件。")
        return
    print("7. 复判台改标签，加入下周训练集。工位模型保持不变。")
    for record in station.review_queue:
        record.note = "人工改标签后回训练机"
        print(f"   件 {record.track_id} 原判定 {record.label.value}，等待班组长确认。")


def run_shift(station: Station, frames: list[Frame]) -> None:
    print(f"\n=== 当班开始：{station.phase.value} ===")
    assert_optical_frozen(station)
    for frame in frames:
        inspect_frame(station, frame)
    review_and_queue_retrain(station)
    print(
        f"当班结束：判定 {len(station.records)} 件，"
        f"其中复判 {len(station.review_queue)} 件。\n"
    )


def demo() -> None:
    # 三件零件。件 1 跨两帧，跟踪只能计一次。
    frames = [
        Frame(1, [Box(Label.GOOD, 0.96, track_id=1)]),
        Frame(2, [Box(Label.GOOD, 0.95, track_id=1)]),
        Frame(3, [Box(Label.HALF_RING, 0.91, track_id=2)]),
        Frame(4, [Box(Label.NO_PLASTIC, 0.62, track_id=3)]),
    ]

    shadow = Station(phase=Phase.SHADOW, optical_frozen=True)
    run_shift(shadow, frames)

    auto = Station(phase=Phase.AUTO, optical_frozen=True)
    run_shift(auto, frames)

    print("=== 心跳丢失 ===")
    dead = Station(phase=Phase.AUTO, optical_frozen=True, heartbeat_ok=False)
    assert_optical_frozen(dead)
    inspect_frame(dead, Frame(1, [Box(Label.GOOD, 0.99, track_id=9)]))
    print("该件未放行。传送带策略是停带，不是放行。")


if __name__ == "__main__":
    demo()
