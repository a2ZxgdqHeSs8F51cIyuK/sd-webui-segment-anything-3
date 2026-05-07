function samGetRealCoordinate(image, x1, y1) {
    if (!image) return [0, 0];
    if (image.naturalHeight * (image.width / image.naturalWidth) <= image.height) {
        const scale = image.naturalWidth / image.width;
        const zero_point = (image.height - image.naturalHeight / scale) / 2;
        const x = x1 * scale;
        const y = (y1 - zero_point) * scale;
        return [x, y];
    } else {
        const scale = image.naturalHeight / image.height;
        const zero_point = (image.width - image.naturalWidth / scale) / 2;
        const x = (x1 - zero_point) * scale;
        const y = y1 * scale;
        return [x, y];
    }
}

function switchToInpaintUpload() {
    switch_to_img2img_tab(4);
    return arguments;
}

function samTabPrefix() {
    const tabs = gradioApp().querySelector('#tabs');
    if (tabs) {
        const buttons = tabs.querySelectorAll('button');
        if (buttons) {
            if (buttons[0].className.includes("selected")) {
                return "txt2img_sam_";
            } else if (buttons[1].className.includes("selected")) {
                return "img2img_sam_";
            }
        }
    }
    return "_sam_";
}

function samImmediatelyGenerate() {
    const runButton = gradioApp().getElementById(samTabPrefix() + "run_button");
    if (runButton && runButton.style.display !== "none") {
        runButton.click();
    }
}

function samIsRealTimePreview() {
    const realtime_preview = gradioApp().querySelector(
        "#" + samTabPrefix() + "realtime_preview_checkbox input[type='checkbox']"
    );
    return realtime_preview && realtime_preview.checked;
}

function samCreateDot(sam_image, image, coord, label) {
    const x = coord.x;
    const y = coord.y;
    const realCoord = samGetRealCoordinate(image, coord.x, coord.y);
    if (realCoord[0] >= 0 && realCoord[0] <= image.naturalWidth &&
        realCoord[1] >= 0 && realCoord[1] <= image.naturalHeight) {
        const isPositive = label == (samTabPrefix() + "positive");
        const circle = document.createElement("div");
        circle.style.position = "absolute";
        circle.style.width = "10px";
        circle.style.height = "10px";
        circle.style.borderRadius = "50%";
        circle.style.left = x + "px";
        circle.style.top = y + "px";
        circle.className = label;
        circle.style.backgroundColor = isPositive ? "black" : "red";
        circle.title = (isPositive ? "positive" : "negative") + " point label, left click it to cancel.";
        sam_image.appendChild(circle);
        circle.addEventListener("click", e => {
            e.stopPropagation();
            circle.remove();
            if (gradioApp().querySelectorAll("." + samTabPrefix() + "positive").length != 0 ||
                gradioApp().querySelectorAll("." + samTabPrefix() + "negative").length != 0) {
                if (samIsRealTimePreview()) {
                    samImmediatelyGenerate();
                }
            }
        });
        if (samIsRealTimePreview()) {
            samImmediatelyGenerate();
        }
    }
}

function samRemoveDots() {
    const sam_image = gradioApp().getElementById(samTabPrefix() + "input_image");
    if (sam_image) {
        ["." + samTabPrefix() + "positive", "." + samTabPrefix() + "negative"].forEach(cls => {
            const dots = sam_image.querySelectorAll(cls);
            dots.forEach(dot => {
                dot.remove();
            });
        });
    }
    return arguments;
}

function create_submit_sam_args(args) {
    // Only copy the arguments, don't nullify anything
    const res = [];
    for (let i = 0; i < args.length; i++) {
        res.push(args[i]);
    }
    return res;
}

function submit_dino() {
    const res = [];
    for (let i = 0; i < arguments.length; i++) {
        res.push(arguments[i]);
    }
    // clear last two placeholders if needed (adjust indices if your Dino usage differs)
    if (res.length >= 5) res[res.length - 2] = null;
    if (res.length >= 6) res[res.length - 1] = null;
    return res;
}

function submit_sam() {
    const res = create_submit_sam_args(arguments); // length = 7
    let positive_points = [];
    let negative_points = [];
    const sam_image = gradioApp().getElementById(samTabPrefix() + "input_image");
    const image = sam_image ? sam_image.querySelector('img') : null;

    // Process click dots only if an image is loaded in the SAM panel
    if (image) {
        const classes = ["." + samTabPrefix() + "positive", "." + samTabPrefix() + "negative"];
        classes.forEach(cls => {
            const dots = sam_image.querySelectorAll(cls);
            dots.forEach(dot => {
                const width = parseFloat(dot.style["left"]);
                const height = parseFloat(dot.style["top"]);
                if (cls == "." + samTabPrefix() + "positive") {
                    positive_points.push(samGetRealCoordinate(image, width, height));
                } else {
                    negative_points.push(samGetRealCoordinate(image, width, height));
                }
            });
        });
    }

    // res[2] and res[3] are the dummy textboxes for positive/negative points
    res[2] = positive_points;
    res[3] = negative_points;

    // Automatic fallback for img2img inpaint image
    if (samTabPrefix().startsWith("img2img") && res.length >= 7) {
        const fileInput = sam_image ? sam_image.querySelector('input[type="file"]') : null;
        const hasImage = fileInput && fileInput.files && fileInput.files.length > 0;
        if (!hasImage) {
            // Forge Neo inpaint image – try several common selectors
            const selectors = [
                '#img2img_image img',        // standard
                '#img_inpaint_base img',     // legacy
                '#img2img_image canvas',     // if shown as a canvas (grab from canvas)
                '#img2img_image > div > img' // fallback
            ];
            let inpaintImage = null;
            for (const sel of selectors) {
                inpaintImage = gradioApp().querySelector(sel);
                if (inpaintImage) break;
            }

            // If it's a canvas, we can still use it directly
            if (!inpaintImage) {
                // Last resort: find any img inside the inpaint tab area
                const inpaintTab = gradioApp().querySelector('#tab_img2img');
                if (inpaintTab) inpaintImage = inpaintTab.querySelector('img');
            }

            if (inpaintImage && (inpaintImage.tagName === 'IMG' || inpaintImage.tagName === 'CANVAS')) {
                try {
                    let dataUrl;
                    if (inpaintImage.tagName === 'CANVAS') {
                        dataUrl = inpaintImage.toDataURL('image/png');
                    } else if (inpaintImage.complete) {
                        const canvas = document.createElement('canvas');
                        canvas.width = inpaintImage.naturalWidth;
                        canvas.height = inpaintImage.naturalHeight;
                        const ctx = canvas.getContext('2d');
                        ctx.drawImage(inpaintImage, 0, 0);
                        dataUrl = canvas.toDataURL('image/png');
                    }
                    if (dataUrl) {
                        res[6] = dataUrl;  // index 6 = hidden textbox
                    }
                } catch (e) {
                    console.warn('SAM fallback image copy failed:', e);
                }
            }
        }
    }

    return res;
}

samPrevImg = {
    "txt2img_sam_": null,
    "img2img_sam_": null,
};

onUiUpdate(() => {
    const sam_image = gradioApp().getElementById(samTabPrefix() + "input_image");
    if (sam_image) {
        const image = sam_image.querySelector('img');
        if (image && samPrevImg[samTabPrefix()] != image.src) {
            samRemoveDots();
            samPrevImg[samTabPrefix()] = image.src;

            image.addEventListener("click", event => {
                const rect = event.target.getBoundingClientRect();
                const x = event.clientX - rect.left;
                const y = event.clientY - rect.top;
                samCreateDot(sam_image, event.target, { x, y }, samTabPrefix() + "positive");
            });

            image.addEventListener("contextmenu", event => {
                event.preventDefault();
                const rect = event.target.getBoundingClientRect();
                const x = event.clientX - rect.left;
                const y = event.clientY - rect.top;
                samCreateDot(sam_image, event.target, { x, y }, samTabPrefix() + "negative");
            });

            const observer = new MutationObserver(mutations => {
                mutations.forEach(mutation => {
                    if (mutation.type === 'attributes' && mutation.attributeName === 'src' && mutation.target === image) {
                        samRemoveDots();
                        samPrevImg[samTabPrefix()] = image.src;
                    }
                });
            });

            observer.observe(image, { attributes: true });
        } else if (!image) {
            samRemoveDots();
            samPrevImg[samTabPrefix()] = null;
        }
    }
});