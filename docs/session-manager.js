class SessionManager {
  constructor() {
    const self = this;

    this.websocket = null;

    this.mouseButton = -1;
    this.theta = 0;
    this.phi = 0;
    this.startX;
    this.startY;
    this.moveX = 0;
    this.moveY = 0;

    this.canvas = document.getElementById("canvas");
    this.canvas.addEventListener("contextmenu", function (e) {
      e.preventDefault();
    });
    this.canvas.addEventListener("pointerdown", function (e) {
      e.preventDefault();
      if (self.websocket.readyState !== 1) return;
      self.mouseButton = e.button;
      self.startX = e.clientX;
      self.startY = e.clientY;
    });
    this.canvas.addEventListener("pointermove", function (e) {
      e.preventDefault();
      if (self.websocket.readyState !== 1) return;
      switch (self.mouseButton) {
        case 0: // 左ボタンが押下されている状態
          self.theta += (e.clientX - self.startX) * 0.01;
          self.phi += (e.clientY - self.startY) * 0.01;
          self.startX = e.clientX;
          self.startY = e.clientY;
          if (self.websocket) {
            self.websocket.send(
              JSON.stringify({
                theta: self.theta,
                phi: self.phi,
              })
            );
          }
          break;
        case 2: // 右ボタンが押下されている状態
          self.moveX -= (e.clientX - self.startX) * 0.01;
          self.moveY += (e.clientY - self.startY) * 0.01;
          self.startX = e.clientX;
          self.startY = e.clientY;
          if (self.websocket) {
            self.websocket.send(
              JSON.stringify({
                moveX: self.moveX,
                moveY: self.moveY,
              })
            );
          }
          break;
      }
    });
    this.canvas.addEventListener("pointerup", function (e) {
      e.preventDefault();
      self.mouseButton = -1;
    });

    this.maxSpp = document.getElementById("max-spp");
    this.maxSpp.addEventListener("input", function () {
      self.maxSpp.value = self.maxSpp.value.replace(/[^0-9]+/i, "");
    });
  }

  init_websocket() {
    return new Promise((resolve) => {
      const url = document.getElementById("endpoint").value;
      const websocket = new WebSocket(url);
      websocket.onopen = () => {
        console.log("open");
        resolve(websocket);
      };
      websocket.onclose = () => {
        console.log("close");
      };
      websocket.onmessage = (message) => {
        message.data
          .slice(0, 4)
          .text()
          .then((header) => {
            switch (header) {
              // レンダリング結果画像を canvas に表示
              case "0000": {
                createImageBitmap(message.data.slice(4)).then((bitmap) => {
                  this.canvas.width = bitmap.width;
                  this.canvas.height = bitmap.height;
                  this.canvas.getContext("2d").drawImage(bitmap, 0, 0);
                });
                break;
              }
              // 現在の1画素あたりのサンプル数をフォームに表示
              case "0001": {
                message.data
                  .slice(4)
                  .arrayBuffer()
                  .then((sample) => {
                    document.getElementById("current-spp").value =
                      sample.byteLength;
                  });
                break;
              }
            }
          });
      };
    });
  }

  start() {
    console.log("start");
    if (this.websocket && this.websocket.readyState === 1) {
      // WebSocket が初期化され，コネクションが開かれている場合
      this.websocket.send(
        JSON.stringify({
          theta: this.theta,
          phi: this.phi,
          moveX: this.moveX,
          moveY: this.moveY,
          maxSpp: this.maxSpp.value,
        })
      );
    } else {
      // WebSocket が初期化されていないか，コネクションが開かれていない場合
      this.init_websocket().then((ws) => {
        this.websocket = ws;
        this.websocket.send(
          JSON.stringify({
            theta: this.theta,
            phi: this.phi,
            moveX: this.moveX,
            moveY: this.moveY,
            maxSpp: this.maxSpp.value,
          })
        );
      });
    }
  }

  end() {
    if (this.websocket) {
      console.log("end");
      this.websocket.close();
      this.websocket = null;
    }
  }

  save() {
    const aTag = document.createElement("a");
    const dataURL = this.canvas.toDataURL("image/png");
    const samplePerPixel = document.getElementById("current-spp").value;
    aTag.setAttribute(
      "href",
      dataURL.replace(/^data:image\/png/, "data:application/octet-stream")
    );
    aTag.setAttribute("download", "image_" + samplePerPixel + "_spp.png");
    aTag.click();
  }
}

const session = new SessionManager();
