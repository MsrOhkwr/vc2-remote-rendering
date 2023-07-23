from ping3 import ping
from pyngrok import conf, ngrok


class Tunnel:
    # Ngrok がサポートするリージョンのリスト
    # cf.) https://ngrok.com/docs/ngrok-agent/config/#region
    region_list = ["us", "eu", "ap", "au", "sa", "jp", "in"]

    def __init__(self):
        self.region_priority_list = Tunnel.region_list
        self.auth_token = None

    def install_auth_token(self, auth_token=None):
        if auth_token is None:
            self.auth_token = input()
        else:
            self.auth_token = auth_token

    def calc_region_priority(self):
        key_list = []
        for region in Tunnel.region_list:
            public_url = self.get_public_url(region=region)
            key = float("inf")
            key = ping(public_url.replace("tcp://", "").split(":")[0])
            ngrok.disconnect(public_url)
            ngrok.kill()
            key_list.append(key)

        self.region_priority_list, _ = zip(
            *sorted(zip(Tunnel.region_list, key_list), key=lambda x: x[1])
        )

        print(self.region_priority_list, _)

    def get_public_url(self, port=80, region=None):
        if region == None:
            region = self.region_priority_list[0]
        pyngrok_config = conf.PyngrokConfig(region=region, auth_token=self.auth_token)
        return ngrok.connect(
            port, proto="tcp", pyngrok_config=pyngrok_config
        ).public_url