import yaml
import json

routers = {
    "master": "172.10.0.1/24",
    "P" : "172.16.3.0/24",
    "Q" : "192.168.10.0/24",
    "R" : "172.12.0.0/16",
    "S" : "",
    "T" : "",
    "U" : "10.85.8.0/24",
    "V" : "10.100.0.0/16"
}
link_address = {
    "PQ": "10.50.1.0/24",
    "QS": "10.51.1.0/24",
    "QV": "10.52.1.0/24",
    "RP": "10.53.1.0/24",
    "RU": "10.54.1.0/24",
    "SR": "10.55.1.0/24",
    "SV": "10.56.1.0/24",
    "TQ": "10.57.1.0/24",
    "US": "10.58.1.0/24",
    "VT": "10.59.1.0/24",
    "VU": "10.60.1.0/24"    
}
devices = {
    'P': {
        'Q': 'P-eth3',
        'R': 'P-eth4'
    },
    'Q': {
        'P': 'Q-eth1',
        'S': 'Q-eth2',
        'V': 'Q-eth3',
        'T': 'Q-eth4'
    },
    'R': {
        'P': 'R-eth1',
        'U': 'R-eth2',
        'S': 'R-eth3'
    },
    'S': {
        'Q': 'S-eth0',
        'R': 'S-eth1',
        'V': 'S-eth2',
        'U': 'S-eth3'
    },
    'T': {
        'Q': 'T-eth0',
        'V': 'T-eth1'
    },
    'V': {
        'Q': 'V-eth1',
        'S': 'V-eth2',
        'T': 'V-eth3', 
        'U': 'V-eth4'
    },
    'U': {
        'R': 'U-eth2',
        'S': 'U-eth3',
        'V': 'U-eth4'
    }
}

def CreateRoutes(route):
    # # grab first node and then set to the rest of the routes
    # firstNode, route = route[0], route[1:]
    # print(route[:-1])
    # firstNodeRoutes = []
    # for currentNode in route[:-1]:
    #     for nextNode in route
    routes = {

    }
    routes = []
    for i in range(len(route) - 1):
        curr = route[i]
        currRoutes = {'router': curr,
                      'entries': []}
        nex = route[i + 1]
        dev = devices[curr][nex]
        link = 0
        route_string = '{} via {} dev {}'.format('default', link, dev)
        currRoutes['entries'].append(route_string)
        try:
            link = link_address['{}{}'.format(curr, nex)]
            link = list(link.split('/')[0])
            link[-1] = '2'
            link = ''.join(link)
            
        except:
            link = list(link_address['{}{}'.format(nex, curr)].split('/')[0])
            link[-1] = '1'
            link = ''.join(link)
        # print('i:{}'.format(curr))

        for j in range(i + 1, len(route)):
            # print('j: {}'.format(nex))
            # {address} via {link} dev {device}
            nex = route[j]
            if nex == 'S' or nex == 'T':
                continue
            route_string = '{} via {} dev {}'.format(routers[nex], link, dev)
            currRoutes['entries'].append(route_string)
            # print(route_string)
        if len(currRoutes['entries']):
            routes.append(currRoutes)

    return routes


if __name__ == '__main__':

    # with open('/home/alex/CompNW_PA3_Scaffolding/hw1prob5_topo.yaml', 'r') as f:
    #     topo = yaml.safe_load(f)
    #     print(json.dumps(topo, indent=4))

    routes = CreateRoutes(['P', 'Q', 'S', 'R', 'U', 'V', 'T'])
    print(routes)
    
    topo = {}
    with open('./empty_topo.yaml', 'r') as f:
        topo = yaml.safe_load(f)
    
    topo['topo']['routes'] = routes

    with open('testing.yaml', 'w') as f:
        yaml.dump(topo, f)
        # f.write(json.dumps(topo, indent=4))