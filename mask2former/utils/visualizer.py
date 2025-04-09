try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import colorsys
import math
import numpy as np
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import pyplot as plt
import matplotlib.colors as mc
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.lines import Line2D

__SMALL_OBJECT_AREA_THRESH__ = 1000


def mask_area(mask):
    """
    Args:
        mask (ndarray): a binary mask of shape (H, W).
    Returns:
        the area of the mask.
    """
    return mask.astype("bool").sum()


def mask2contour(binary_mask):
    contours = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contour = max(contours, key=cv2.contourArea)
    return contour


def mask2box(mask, format="xywh"):
    cnt = mask2contour(mask)
    x, y, w, h = cv2.boundingRect(cnt)
    if format == "xywh":
        return [x, y, w, h]
    elif format == "xyxy":
        return [x, y, x + w, y + h]


def mask2contours(binary_mask):
    contours = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    return contours


def mask_to_polygons(mask):
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
    mask = np.ascontiguousarray(
        mask
    )  # some versions of cv2 does not support incontiguous arr
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]
    # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
    # We add 0.5 to turn them into real-value coordinate space. A better solution
    # would be to first +0.5 and then dilate the returned polygon by 0.5.
    res = [x + 0.5 for x in res if len(x) >= 6]
    return res, has_holes


def change_color_brightness(color, brightness_factor):
    """
    Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
    less or more saturation than the original color.

    Args:
        color: color of the polygon. Refer to `matplotlib.colors` for a full list of
            formats that are accepted.
        brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
            0 will correspond to no change, a factor in [-1.0, 0) range will result in
            a darker color and a factor in (0, 1.0] range will result in a lighter color.

    Returns:
        modified_color (tuple[double]): a tuple containing the RGB values of the
            modified color. Each value in the tuple is in the [0.0, 1.0] range.
    """
    assert brightness_factor >= -1.0 and brightness_factor <= 1.0
    color = mc.to_rgb(color)
    polygon_color = colorsys.rgb_to_hls(*mc.to_rgb(color))
    modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
    modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
    modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
    modified_color = colorsys.hls_to_rgb(
        polygon_color[0], modified_lightness, polygon_color[2]
    )
    return tuple(np.clip(modified_color, 0.0, 1.0))


class Visualizer:
    def __init__(self, image, scale=1.0, cmap=None):
        self.original_image = image
        self.scale = scale
        self.fig = plt.figure(frameon=None)
        self.dpi = self.fig.get_dpi()
        self.height, self.width = image.shape[0], image.shape[1]
        self._default_font_size = max(
            np.sqrt(self.height * self.width) // 90, 10 // scale
        )
        self.fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis("off")
        self.reset_image()
        self.cmap = (
            cmap
            if cmap is not None
            else (
                "#a6cee3",
                "#1f78b4",
                "#b2df8a",
                "#33a02c",
                "#fb9a99",
                "#e31a1c",
                "#fdbf6f",
                "#ff7f00",
                "#cab2d6",
                "#6a3d9a",
                "#ffff99",
                "#b15928",
                "#8dd3c7",
                "#333333",
                "#bebada",
                "#fb8072",
                "#80b1d3",
                "#fdb462",
                "#b3de69",
                "#fccde5",
                "#d9d9d9",
                "#bc80bd",
                "#ccebc5",
                "#ffed6f",
            )
        )

    def get_output(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")

    def reset_image(self):
        self.ax.imshow(
            self.original_image,
            extent=(0, self.width, self.height, 0),
            interpolation="nearest",
        )

    def draw_polygon(self, polygon, color, edge_color, alpha=0.5):
        if edge_color is None:
            # make edge color darker than the polygon color
            if alpha > 0.8:
                edge_color = change_color_brightness(color, brightness_factor=-0.7)
            else:
                edge_color = color
        edge_color = mc.to_rgb(edge_color) + (1,)
        codes = [Path.MOVETO] + [Path.LINETO] * (len(polygon) - 2) + [Path.CLOSEPOLY]
        path = Path(polygon, codes)
        patch = PathPatch(
            path,
            fill=True,
            facecolor=mc.to_rgb(color) + (alpha,),
            edgecolor=edge_color,
        )
        self.ax.add_patch(patch)
        return self

    def draw_masks(self, masks):
        for i, mask in enumerate(masks):
            color = self.cmap[i % len(self.cmap)]
            polygons, _ = mask_to_polygons(mask)
            self.draw_polygon(polygons.reshape(-1, 2), color, color)
        return self

    def draw_box_xyxy(self, bbox, edge_color, line_style="-", alpha=0.5):
        x1, y1, x2, y2 = bbox
        self.ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=edge_color,
                linewidth=2,
                linestyle=line_style,
                alpha=alpha,
            )
        )
        return self

    def draw_box_xywh(self, bbox, edge_color, line_style="-", alpha=0.5):
        x, y, w, h = bbox
        self.ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                fill=False,
                edgecolor=edge_color,
                linewidth=2,
                linestyle=line_style,
                alpha=alpha,
            )
        )
        return self

    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        font_color="g",
        horizontal_alignment="center",
        rotation=0,
        **kwargs,
    ):
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mc.to_rgb(font_color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.ax.text(
            x,
            y,
            text,
            size=font_size * self.scale,
            family="sans-serif",
            color=color,
            horizontalalignment=horizontal_alignment,
            zorder=10,
            rotation=rotation,
            bbox={"facecolor": "black", "alpha": 0.5, "pad": 0.7, "linewidth": 0},
            **kwargs,
        )
        return self

    def draw_line(self, x_data, y_data, color, linestyle="-", linewidth=None):
        """
        Args:
            x_data (list[int]): a list containing x values of all the points being drawn.
                Length of list should match the length of y_data.
            y_data (list[int]): a list containing y values of all the points being drawn.
                Length of list should match the length of x_data.
            color: color of the line. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            linestyle: style of the line. Refer to `matplotlib.lines.Line2D`
                for a full list of formats that are accepted.
            linewidth (float or None): width of the line. When it's None,
                a default value will be computed and used.

        Returns:
            output (VisImage): image object with line drawn.
        """
        if linewidth is None:
            linewidth = self._default_font_size / 3
        linewidth = max(linewidth, 1)
        self.ax.add_line(
            Line2D(
                x_data,
                y_data,
                linewidth=linewidth * self.scale,
                color=color,
                linestyle=linestyle,
            )
        )
        return self

    def draw_rotated_box_with_label(
        self, rotated_box, alpha=0.5, edge_color="g", line_style="-", label=None
    ):
        """
        Draw a rotated box with label on its top-left corner.

        Args:
            rotated_box (tuple): a tuple containing (cnt_x, cnt_y, w, h, angle),
                where cnt_x and cnt_y are the center coordinates of the box.
                w and h are the width and height of the box. angle represents how
                many degrees the box is rotated CCW with regard to the 0-degree box.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.
            label (string): label for rotated box. It will not be rendered when set to None.

        Returns:
            output (VisImage): image object with box drawn.
        """
        cnt_x, cnt_y, w, h, angle = rotated_box
        area = w * h
        # use thinner lines when the box is small
        linewidth = self._default_font_size / (
            6 if area < __SMALL_OBJECT_AREA_THRESH__ * self.scale else 3
        )

        theta = angle * math.pi / 180.0
        c = math.cos(theta)
        s = math.sin(theta)
        rect = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]
        # x: left->right ; y: top->down
        rotated_rect = [
            (cnt_x - xx * c + yy * s, cnt_y - xx * s - yy * c) for (xx, yy) in rect
        ]
        for k in range(4):
            j = (k + 1) % 4
            self.draw_line(
                [rotated_rect[k][0], rotated_rect[j][0]],
                [rotated_rect[k][1], rotated_rect[j][1]],
                color=edge_color,
                linestyle="--" if k == 1 else line_style,
                linewidth=linewidth,
            )

        if label is not None:
            text_pos = rotated_rect[1]  # topleft corner

            height_ratio = h / np.sqrt(self.height * self.width)
            label_color = self._change_color_brightness(
                edge_color, brightness_factor=0.7
            )
            font_size = (
                np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                * 0.5
                * self._default_font_size
            )
            self.draw_text(
                label, text_pos, color=label_color, font_size=font_size, rotation=angle
            )

        return self

    def draw_box_roate(self, rotated_box, edge_color, line_style="-"):
        """
        Draw a rotated box with label on its top-left corner.

        Args:
            rotated_box (tuple): a tuple containing (cnt_x, cnt_y, w, h, angle),
                where cnt_x and cnt_y are the center coordinates of the box.
                w and h are the width and height of the box. angle represents how
                many degrees the box is rotated CCW with regard to the 0-degree box.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.
            label (string): label for rotated box. It will not be rendered when set to None.

        Returns:
            output (VisImage): image object with box drawn.
        """
        # print(rotated_box)
        cnt_x, cnt_y, w, h, angle = rotated_box
        area = w * h
        # use thinner lines when the box is small
        linewidth = self._default_font_size / (
            20 if area < __SMALL_OBJECT_AREA_THRESH__ * self.scale else 10
        )

        theta = angle * math.pi / 180.0
        c = math.cos(theta)
        s = math.sin(theta)
        rect = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]
        # x: left->right ; y: top->down
        # cw rotate
        rotated_rect = [
            (cnt_x - xx * c + yy * s, cnt_y - xx * s - yy * c) for (xx, yy) in rect
        ]
        for k in range(4):
            j = (k + 1) % 4
            self.draw_line(
                [rotated_rect[k][0], rotated_rect[j][0]],
                [rotated_rect[k][1], rotated_rect[j][1]],
                color=edge_color,
                linestyle=line_style,
                linewidth=linewidth,
            )
        return self

    def overlay_instances(
        self,
        *,
        labels=None,
        masks=None,
        boxes=None,
        scores=None,
        box_format: Literal["rotate", "xyxy", "xywh"] = "xyxy",
        label_mapping=lambda x: x,
        text_mapping=lambda label, score: f"{label if label else ''}{'_' if (label and score) else ''}{'%.2f' % score if score else ''}",
    ):
        num_instances = 0
        if boxes is not None:
            num_instances = len(boxes)
        if masks is not None:
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)

        if labels is not None and scores is not None:
            assert len(labels) == num_instances and len(scores) == num_instances
        elif labels is not None and scores is None:
            scores = np.zeros_like(labels, dtype=np.bool8)
        elif labels is None and scores is not None:
            labels = np.zeros_like(scores, dtype=np.bool8)
        if num_instances == 0:
            return self
        areas = None
        if boxes is not None:
            if box_format == "rotate":
                areas = boxes[:, 2] * boxes[:, 3]
            else:
                areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([mask_area(x) for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            scores = scores[sorted_idxs] if scores is not None else None

        for i in range(num_instances):
            color = self.cmap[i % len(self.cmap)]

            # draw box
            if boxes is not None:
                if box_format == "xyxy":
                    self.draw_box_xyxy(
                        boxes[i], change_color_brightness(color, brightness_factor=-0.2)
                    )
                elif box_format == "xywh":
                    self.draw_box_xywh(
                        boxes[i], change_color_brightness(color, brightness_factor=-0.2)
                    )
                elif box_format == "rotate":
                    self.draw_box_roate(
                        boxes[i], change_color_brightness(color, brightness_factor=-0.2)
                    )
                else:
                    raise ValueError("Unknown box format: {}".format(box_format))

            # draw mask
            if masks is not None:
                polygons, _ = mask_to_polygons(masks[i])
                for polygon in polygons:
                    self.draw_polygon(
                        polygon.reshape(-1, 2),
                        color,
                        change_color_brightness(color, brightness_factor=-0.5),
                    )

            # draw text
            if labels is None or scores is None:
                continue
            else:
                if boxes is not None:
                    if box_format == "xywh":
                        _x, _y, _w, _h = boxes[i]
                        x1, y1, x2, y2 = _x, _y, _x + _w, _y + _h
                        text_pos = (x1, y1)
                        horizontal_align = "left"
                    elif box_format == "xyxy":
                        x1, y1, x2, y2 = boxes[i]
                        text_pos = (x1, y1)
                        horizontal_align = "left"
                    elif box_format == "rotate":
                        cx, cy, r_w, r_h, angle = boxes[i]
                        theta = angle * math.pi / 180.0
                        points = [(-r_w / 2, -r_h / 2), (r_w / 2, r_h / 2)]
                        (x1, y1), (x2, y2) = [
                            (
                                cx - xx * math.cos(theta) + yy * math.sin(theta),
                                cy - xx * math.sin(theta) - yy * math.cos(theta),
                            )
                            for (xx, yy) in points
                        ]
                        text_pos = (x1, y1)
                        horizontal_align = "left"
                    else:
                        raise ValueError("Unknown box format: {}".format(box_format))

                elif masks is not None:
                    text_pos = np.median(masks[i].nonzero(), axis=1)[::-1]
                    x1, y1, x2, y2 = mask2box(masks[i], format="xyxy")
                    horizontal_align = "center"

                else:
                    continue

                if box_format == "rotate":
                    text_pos = (x1, y1)
                    height_ratio = r_h / np.sqrt(self.height * self.width)
                    font_size = (
                        np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                        * 0.5
                        * self._default_font_size
                    )
                else:
                    instance_area = (x2 - x1) * (y2 - y1)
                    if (
                        instance_area < __SMALL_OBJECT_AREA_THRESH__ * self.scale
                        or y2 - y1 < 40 * self.scale
                    ):
                        if y1 > self.height - 5:
                            text_pos = (x2, y1)
                        else:
                            text_pos = (x1, y2)

                    height_ratio = (y2 - y1) / np.sqrt(self.height * self.width)

                    font_size = (
                        np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                        * 0.5
                        * self._default_font_size
                    )
                self.draw_text(
                    text_mapping(label_mapping(labels[i]), scores[i]),
                    text_pos,
                    font_size=font_size,
                    font_color=change_color_brightness(color, brightness_factor=0.5),
                    horizontal_alignment=horizontal_align,
                )
        return self