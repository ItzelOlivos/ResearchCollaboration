import matplotlib.pyplot as plt
import urllib3
import json
import igraph as ig
import numpy as np
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import sys
import argparse
import warnings
import csv
warnings.filterwarnings("ignore", category=DeprecationWarning)
# import re
# import io
# import networkx as nx
# import csv
# import pandas as pd
# import plotly.express as px
# from selenium import webdriver

http = urllib3.PoolManager()

# STEP 0. Clicking on websites:
"""
driver = webdriver.Chrome('/Users/ico1/PycharmProjects/Portfolio/ScholarNetwork/chromedriver')  # Optional argument, if not specified will search path.
driver.get('http://www.google.com/');
time.sleep(5) # Let the user actually see something!
search_box = driver.find_element_by_name('q')
search_box.send_keys('ChromeDriver')
search_box.submit()
time.sleep(5) # Let the user actually see something!
driver.quit()
"""


# ======================= Auxiliary steps ==============================================
def create_empty_graph_record(filename):
    data = dict.fromkeys(['nodes', 'links'])
    data['nodes'] = []
    data['links'] = []

    with open(filename, 'w') as outfile:
        json.dump(data, outfile, sort_keys=True, indent=4)


def save_changes(filename, nodes, links):
    data = dict.fromkeys(['nodes', 'links'])
    data['nodes'] = nodes
    data['links'] = links

    with open(filename, 'w') as outfile:
        json.dump(data, outfile, sort_keys=True, indent=4)


def explore_deeper(filename, next_file):
    with open(filename, 'r') as f:
        json_data = json.load(f)

    new_ids = []
    for record in json_data['links']:
        new_ids.append(record['source'])
        new_ids += record['target']

    new_ids = "\n".join(list(set(new_ids)))

    with open(next_file, "w") as my_file:
        my_file.write(new_ids)


# ======================= Step 1. Given a scholar id, retrieve data =======================
def retrieve_user_profile(scholar_id, nodes, links, dig_deeper):
    success = False
    try:
        url = 'https://scholar.google.com/citations?user=' + scholar_id + '&hl=en&oi=ao'
        response = http.request('GET', url)
        soup = BeautifulSoup(response.data)
        # print(soup.prettify())
    except:
        print(f'Warning: Profile {scholar_id} could not been retrieved')
        return nodes, links

    name = soup.find('div', {'id':'gsc_prf_in'})
    if name is not None:
        name = name.get_text()
    else:
        name = "?"

    affiliation = soup.find('div', {'class':'gsc_prf_il'})
    if affiliation is not None:
        affiliation = affiliation.get_text()
    else:
        affiliation = "?"

    homepage = soup.find('div', {'id':'gsc_prf_ivh'})
    if homepage is not None:
        homepage = homepage.find('a')
        if homepage is not None:
            homepage = homepage.get('href')
        else:
            homepage = "?"
    else:
        homepage = "?"

    try:
        most_cited_paper = soup.find('tbody', {'id': 'gsc_a_b'})
        best_work = most_cited_paper.find("a", class_="gsc_a_at").get_text() + ". Cited by " + most_cited_paper.find(
            "a", class_="gsc_a_ac gs_ibl").get_text() + " in " + most_cited_paper.find("span",
                                                                                       class_="gsc_a_h gsc_a_hc gs_ibl").get_text()
        best_work_cites = most_cited_paper.find("a", class_="gsc_a_ac gs_ibl").get_text()
        best_work_year = most_cited_paper.find("span", class_="gsc_a_h gsc_a_hc gs_ibl").get_text()
    except:
        best_work = "?"
        best_work_cites = "?"
        best_work_year = "?"

    try:
        keywords = [k.get_text().lower() for k in soup.find('div', {'id': 'gsc_prf_int'}).find_all('a')]
    except:
        keywords = []

    new_node = {'int_id': len(nodes), 'user_id': scholar_id, 'name': name,
                'most_cited_paper': best_work, 'mcp_cites':best_work_cites, 'mcp_year':best_work_year,
                'affiliation': affiliation, 'homepage': homepage, 'keywords': keywords}

    nodes.append(new_node)
    success = True

    if dig_deeper:
        try:
            coauthors = []
            spans = soup.find_all('span', {'class': 'gsc_rsb_a_desc'})
            for span in spans:
                next_url = span.find_all('a', href=True)
                tmp_str = str(next_url[0]["href"])
                tmp_str = tmp_str.split('=', -1)[1]
                next_user_id = tmp_str.split('&', -1)[0]
                coauthors.append(next_user_id)
        except:
            pass

        new_link = {"source": scholar_id, "target": coauthors}
        links.append(new_link)

    return nodes, links, success


# ======================= STEP 2. Adjust links: integers instead of user_ids(str) =======================
def link_by_integer_id(filename):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            links = data['links']
            nodes = data['nodes']
    except:
        print(f"Error. {filename} not found")
        return

    integer_links = []
    for record in links:
        for item in nodes:
            if record['source'] == item["user_id"]:
                id_source = item["int_id"]
                break

        for target in record['target']:
            done = False
            for item in nodes:
                if target == item["user_id"]:
                    id_target = item['int_id']
                    integer_links.append({"source": id_source, "target": id_target})
                    done = True
                    break
            if not done:
                nodes, links, success = retrieve_user_profile(target, nodes, [], False)
                if success:
                    integer_links.append({"source": id_source, "target": nodes[-1]['int_id']})

    # Save changes in integer_links.json
    tmp_links = dict.fromkeys(['links'])
    tmp_links['links'] = integer_links

    with open(filename.replace('.json', '_integer_links.json'), 'w') as outfile:
        json.dump(tmp_links, outfile, sort_keys=True, indent=4)

    save_changes(filename, nodes, links)


# ======================= STEP 3. Create the graph =======================
def creating_graph(filename):
    # chart_studio.tools.set_credentials_file(username='itzelOlivos', api_key='lRr0P5O1rjBxCb2iwiEL')

    with open(filename, 'r') as f:
        json_data = json.load(f)

    N = len(json_data['nodes'])

    with open(filename.replace('.json', '_integer_links.json'), 'r') as f:
        integer_links = json.load(f)

    links = integer_links['links']

    Edges = [(links[k]['source'], links[k]['target']) for k in range(len(links))]
    Vertex = [item['name'] for item in json_data['nodes']]

    G = ig.Graph()
    G.add_vertices(N)
    G.add_edges(Edges)
    G.vs['name'] = Vertex
    G.vs['affiliation'] = [item['affiliation'] for item in json_data['nodes']]
    G.vs['home_page'] = [item['homepage'] for item in json_data['nodes']]
    G.vs['google_profile'] = [item['user_id']for item in json_data['nodes']]
    G.vs['keywords'] = [item['keywords'] for item in json_data['nodes']]
    G.vs['most_cited_paper'] = [item['most_cited_paper'] for item in json_data['nodes']]
    G.vs['cited_by'] = [item['mcp_cites'] for item in json_data['nodes']]
    G.vs['most_cited_year'] = [item['mcp_year'] for item in json_data['nodes']]

    return G


# ======================= STEP 4. Compute graph measures =======================
def compute_graph_measures(filename, G):
    d = G.indegree()
    b = G.betweenness()

    graph_measures = dict.fromkeys(['degree', 'betweenness'])
    graph_measures['degree'] = d
    graph_measures['betweenness'] = b

    G.vs['in'] = graph_measures['degree']
    G.vs['between'] = graph_measures['betweenness']

    with open(filename.replace('.json', '_graph_measures.json'), 'w') as outfile:
        json.dump(graph_measures, outfile, sort_keys=True, indent=4)

    return G


# ======================= STEP 5. Show heavy nodes in General Graph =======================
def display_graph(G, top, beginning_outfile):
    Showing = top
    m = sorted(G.vs, key=lambda z: z['between'], reverse=True)

    with open(beginning_outfile + '_nodes.txt', 'w') as f:
        for e in m[:Showing]:
            f.write('%-9s\t Betweenness: %.2f\t in_degree: %i\t \n' % (e['name'], e['between'], e['in']))
            f.write('Afiliation: %s \n' % e['affiliation'])
            f.write('Most cited paper: %s \n' % (e['most_cited_paper']))
            f.write('GS profile: %s \n' % e['google_profile'])
            f.write('Home page: %s \n' % e['home_page'])
            f.write('Kw: %s \n' % ', '.join(e['keywords']))
            f.write('\n')
    f.close()

    # Creating CSV data file:
    header = ['name', 'Betweenness', 'in_degree', 'affiliation', 'home_page', 'google_profile', 'keywords',
              'most_cited_paper', 'cited_by', 'most_cited_year']

    with open(beginning_outfile + '.csv', 'w', encoding='UTF8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        for e in m[:Showing]:
            data = [e['name'], e['between'], e['in'], e['affiliation'], e['home_page'], e['google_profile'], e['keywords'],
                    e['most_cited_paper'], e['cited_by'], e['most_cited_year']]

            writer.writerow(data)

# ======================= STEP 6. Showing the graph =======================
# How can I add options to change param that determines marker size
# How to show info when touching node
# Something is happening here with the edges displayed


def plot_my_graph(G, top):
    m = sorted(G.vs, key=lambda z: z['between'], reverse=True)
    Block = G.subgraph(m[:top])
    # Edges = ig.EdgeSeq(Block)
    Edges = Block.get_edgelist()
    N = Block.vcount()

    labels = []
    affiliation = []
    between_centrality = []

    max_degree = sorted(m, key=lambda z: (z['in']), reverse=True)
    max_degree = max_degree[0]['in']

    affiliations = []
    deflen=30
    for node in m[:top]:
        affiliations.append(node['affiliation'][:deflen])
    set_of_affiliations = list(set(affiliations))

    for node in m[:top]:
        mystr = node['name'] + ': ' + ', '.join(node['keywords'])
        labels.append(mystr)
        tmp = node['affiliation'][:deflen]
        affiliation.append(set_of_affiliations.index(tmp))
        if max_degree != 0:
            between_centrality.append(int(80 * node['in'] / max_degree))
        else:
            between_centrality.append(0)

    layt = Block.layout('kk', dim=3)

    # Coordinates of nodes
    Xn = [layt[k][0] for k in range(N)]  # x-coordinates of nodes
    Yn = [layt[k][1] for k in range(N)]  # y-coordinates
    Zn = [layt[k][2] for k in range(N)]  # z-coordinates
    Xe = []
    Ye = []
    Ze = []

    for e in Edges:
        # Layt is a list of N vertices, each is an x,y,z coordinate
        # e says: connect node e[0] with node e[1]
        Xe += [layt[e[0]][0], layt[e[1]][0]]  # x-coordinates of edge ends
        Ye += [layt[e[0]][1], layt[e[1]][1]]
        Ze += [layt[e[0]][2], layt[e[1]][2]]

    custom_bar = None
    if top < 50:
        custom_bar = dict(title='Affiliations', titleside='top', tickmode='array', tickvals=np.arange(0, len(set_of_affiliations)), ticktext=set_of_affiliations, ticks='outside')

    trace1 = go.Scatter3d(x=Xe,
                          y=Ye,
                          z=Ze,
                          mode='lines',
                          line=go.Line(color='rgb(125,125,125)', width=1.8),
                          hoverinfo='none'
                          )

    trace2 = go.Scatter3d(x=Xn,
                          y=Yn,
                          z=Zn,
                          mode='markers',
                          name='people',
                          marker=dict(size=between_centrality, #6
                                      color=affiliation,
                                      colorbar=custom_bar,
                                      colorscale='Viridis',
                                      line=go.Line(color='rgb(50,50,50)', width=0.5)
                                      ),
                          text=labels,
                          hoverinfo='text'
                          )

    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )

    layout = go.Layout(
        title=f"Based on Google Scholar profiles, network has {len(m)} nodes. \n Showing top {int(top*100/len(m))}%.",
        width=1000,
        height=700,
        showlegend=False,
        scene=go.Scene(
            xaxis=go.XAxis(axis),
            yaxis=go.YAxis(axis),
            zaxis=go.ZAxis(axis),
        ),
        margin=go.Margin(
            t=100
        ),
        hovermode='closest',
        annotations=go.Annotations([
            go.Annotation(
                showarrow=False,
                text="Size of nodes is based on its betweeness centrality. \n Created by Itzel Olivos-Castillo",
                xref='paper',
                yref='paper',
                x=0,
                y=0.1,
                xanchor='left',
                yanchor='bottom',
                font=go.Font(
                    size=14
                )
            )
        ]),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Zoom in",
                          method="animate",
                          args=[None])])]
    )

    data = go.Data([trace1, trace2])
    return data, layout

    # return go.Figure(data=data, layout=layout)


def run_me(filename, net_file):
    with open(filename) as f:
        lines = f.readlines()

    nodes = []
    links = []

    for idx, line in enumerate(lines):
        scholar_id = line.replace('\n', '')
        [nodes, links, _] = retrieve_user_profile(scholar_id, nodes, links, True)
        print(f"Processing: {scholar_id}")

    create_empty_graph_record(net_file)
    save_changes(net_file, nodes, links)


# ======================= MAIN =======================

if __name__ == "__main__":

    # Format should be: python main.py --t d --p people.txt --e 1 --o minds

    parser = argparse.ArgumentParser(description='Google Scholar Collaboration Network')
    parser.add_argument('--t', type=str, help='computing (c) or displaying? (d)')
    parser.add_argument('--p', type=str, help='path of the txt file listing Google Scholar profiles of interest')
    parser.add_argument('--e', type=str, help='how many hops the scrapper should explore')
    parser.add_argument('--o', type=str, help='first word of output files´ title')

    if len(sys.argv) == 9:
        args = parser.parse_args()

        origin_file = args.p
        depth = int(args.e)
        work = args.t
        beginning_outfile = args.o
        net_file = beginning_outfile + ".json"

        if work == 'c':

            run_me(origin_file, net_file)
            for iteration in range(depth):
                explore_deeper(net_file, 'new_' + origin_file)
                run_me('new_' + origin_file, net_file)

            link_by_integer_id(net_file)

            G = creating_graph(net_file)
            G = compute_graph_measures(net_file, G)
            display_graph(G, -1, beginning_outfile)

            fig = plot_my_graph(G, -1)
            title = beginning_outfile + ".html"
            fig.write_html(title)
        else:
            with open(net_file, 'r') as f:
                json_data = json.load(f)

            N = len(json_data['nodes'])
            print(f"There are {N} nodes in the graph")
            print("Display top d% nodes sorted by its betweenness centrality")
            top = int(int(input("d? ")) * N / 100)
            G = creating_graph(net_file)
            G = compute_graph_measures(net_file, G)
            display_graph(G, top, beginning_outfile)

            data1, layout1 = plot_my_graph(G, int(100 * N / 100))
            data2, layout2 = plot_my_graph(G, int(50 * N / 100))
            data3, layout3 = plot_my_graph(G, int(25 * N / 100))
            data4, layout4 = plot_my_graph(G, int(12 * N / 100))
            data5, layout5 = plot_my_graph(G, int(6 * N / 100))
            data6, layout6 = plot_my_graph(G, int(3 * N / 100))

            frames = [go.Frame(data=data2, layout=layout2),
                      go.Frame(data=data3, layout=layout3),
                      go.Frame(data=data4, layout=layout4),
                      go.Frame(data=data5, layout=layout5),
                      go.Frame(data=data6, layout=layout6)
                      ]

            fig = go.Figure(data=data1, layout=layout1, frames=frames)
            title = beginning_outfile + ".html"
            fig.write_html(title)

    else:
        parser.print_help()





