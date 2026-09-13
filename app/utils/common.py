def parse_rtsp_url(url):
    # RTSP 스키마 제거
    scheme, rest = url.split("://", 1)
    
    # 마지막 @ 를 기준으로 사용자 정보와 나머지 URL 분리
    at_count = rest.count('@')
    if at_count > 0:
        userinfo, rest = rest.rsplit('@', 1)
        # 첫 번째 : 를 기준으로 username과 password 분리
        colon_index = userinfo.find(':')
        if colon_index != -1:
            username = userinfo[:colon_index]
            password = userinfo[colon_index+1:]
        else:
            username, password = userinfo, None
    else:
        username, password = None, None
    
    # 호스트, 포트, 경로 분리
    if '/' in rest:
        host_port, path = rest.split('/', 1)
        path = '/' + path
    else:
        host_port, path = rest, ''
    
    if ':' in host_port:
        host, port = host_port.split(':')
    else:
        host, port = host_port, '554'  # RTSP 기본 포트
    
    return username, password, f"{scheme}://{host}:{port}{path}"