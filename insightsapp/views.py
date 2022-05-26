from unicodedata import numeric
from unittest import result
from django.shortcuts import redirect, render
from django.views import View
import os.path
import pandas as pd
import json
import openpyxl
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
import seaborn as sns

# Create your views here.

df = pd.DataFrame()
suitable_columns =[]
date_columns =[]
numeric_columns =[]


class Home(View):
    def get(self,request):
        return render(request,'insightsapp/home.html')
    def post(self,request):
        file = request.FILES["file"]
        extension = os.path.splitext(file.name)[1]
        if extension == '.xlsx':    
            df1 = pd.read_excel(file)
        elif extension=='.csv':
            df1 = pd.read_csv(file, encoding = 'unicode_escape')
        else:
            return render(request,'insightsapp/home.html',{'err':'File must be excel worksheet or csv'})
        global df
        df = df1.copy()
        present_columns = df.columns.to_list()
        for i in present_columns:
            if df[i].dtypes == 'datetime64[ns]':
                df[i] = df[i].astype(str)
        json_records = df.head(15).reset_index().to_json(orient ='records')
        data = []
        data = json.loads(json_records)
        context = {'data': data,'present_columns':present_columns}
        return render(request, 'insightsapp/home.html', context)



def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i], ha = 'center')



class SingleInsights(View):
    def get(self,request):
        err = request.session.get('err',False)
        if err:
            del(request.session['err'])
        global df
        present_columns = df.columns.to_list()
        global suitable_columns
        global date_columns
        global numeric_columns
        suitable_columns_local =[]
        date_columns_local =[]
        numeric_columns_local = []
        for i in present_columns:
            if df[i].dtypes == 'float64' or df[i].dtypes == 'int64':
                numeric_columns_local.append(i)
            elif df[i].dtypes == 'datetime64[ns]':
                date_columns_local.append(i)
            else:
                try:
                    pd.to_datetime(df[i])
                    date_columns_local.append(i)
                except:
                    try:
                        df[i].astype(float)
                        numeric_columns.append(i)
                    except:
                        total_rows = df.shape[0]
                        i_count = df[i].nunique()
                        if total_rows > 1000:
                            if i_count <= df.shape[0]*0.25:
                                suitable_columns_local.append(i)
                        else:
                            suitable_columns_local.append(i)
        suitable_columns =suitable_columns_local.copy()
        date_columns = date_columns_local.copy()
        numeric_columns = numeric_columns_local.copy()
        print(numeric_columns)

        context = {'suitable_columns':suitable_columns,'date_columns':date_columns,'numeric_columns':numeric_columns,'err':err}
        return render(request, 'insightsapp/insights.html', context)

    def post(self,request):
        insight = request.POST.get('insight')
        parameter = request.POST.get('parameter')
        global df
        if parameter == 'Count':
            if insight in date_columns:
                df[insight] = pd.to_datetime(df[insight])
                insight_results = df.resample('M',on = insight).count()
                insight_results = insight_results.iloc[:,0:1]
                results = insight_results.rename(columns={insight_results.columns[0]:'Count'})
                results = results.to_period('M').reset_index()
                results[insight] = results[insight].dt.strftime("%Y-%b")
                present_columns = results.columns.to_list()
                results.index=results.index.to_series().astype(str)
                results.plot.line(x=insight,y='Count').set_ylabel('Count')
                buf = BytesIO()
                plt.savefig(buf, format='png',dpi = 300, bbox_inches='tight')
                line_plot = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
                buf.close()
                results[insight] = results[insight].astype(str)
                json_records = results.reset_index().to_json(orient ='records')
                data = []
                data = json.loads(json_records)
                max_count = results[results.Count == results.Count.max()]
                json_records_max = max_count.reset_index().to_json(orient = 'records')
                max_data = []
                max_data = json.loads(json_records_max)
                min_count = results[results.Count == results.Count.min()]
                json_records_min = min_count.reset_index().to_json(orient = 'records')
                min_data = []
                min_data = json.loads(json_records_min)
                context = {'data':data,'present_columns':present_columns,'line_plot':line_plot,'max_data':max_data,'min_data':min_data,'insight':insight}
                df[insight] = df[insight].astype(str)
            else:
                insight_results = df.groupby(insight).count().reset_index()
                insight_results = insight_results.iloc[:,0:2]
                results = insight_results.rename(columns={insight_results.columns[1]:'Count'})
                present_columns = results.columns.to_list()
                json_records = results.reset_index().to_json(orient ='records')
                data = []
                data = json.loads(json_records)
                max_count = results[results.Count == results.Count.max()]
                json_records_max = max_count.reset_index().to_json(orient = 'records')
                max_data = []
                max_data = json.loads(json_records_max)
                min_count = results[results.Count == results.Count.min()]
                json_records_min = min_count.reset_index().to_json(orient = 'records')
                min_data = []
                min_data = json.loads(json_records_min)
                context = {'data': data,'present_columns':present_columns,'max_data':max_data,'min_data':min_data,'insight':insight}
                try:
                    results.plot.pie(y='Count',labeldistance=None,figsize=(9,9),autopct= lambda p:'{:.2f}% ({:,.0f})'.format(p,p*df[insight].count()/100))
                    labels = results[insight].tolist()
                    plt.legend(labels,title= 'LEGEND',bbox_to_anchor=(1,1))
                    plt.ylabel(None)
                    buf = BytesIO()
                    plt.savefig(buf, format='png',dpi = 300, bbox_inches='tight')
                    pie_chart = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
                    buf.close()
                    context.update({'pie_chart':pie_chart})
                except:
                    pass
                try:
                    results.plot.bar(x=insight,y='Count',figsize=(9,9))
                    addlabels(results[insight], results['Count'])
                    plt.ylabel('Count')
                    buf = BytesIO()
                    plt.savefig(buf, format='png',dpi = 300, bbox_inches='tight')
                    bar_chart = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
                    buf.close()
                    context.update({'bar_chart':bar_chart})
                except:
                    pass
            return render(request,'insightsapp/insight_results.html',context)
        else:
            if insight in date_columns:
                df[insight] = pd.to_datetime(df[insight])
                insight_results = df.resample('M',on = insight)[parameter].sum()
                insight_results = insight_results.to_frame()
                print(insight_results)
                results = insight_results.rename(columns={insight_results.columns[0]:'Total'})
                print(results)
                results = results.to_period('M').reset_index()
                results[insight] = results[insight].dt.strftime("%Y-%b")
                present_columns = results.columns.to_list()
                results.index=results.index.to_series().astype(str)
                results.plot.line(x=insight,y='Total').set_ylabel('Total')
                buf = BytesIO()
                plt.savefig(buf, format='png',dpi = 300, bbox_inches='tight')
                line_plot = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
                buf.close()
                results[insight] = results[insight].astype(str)
                json_records = results.reset_index().to_json(orient ='records')
                data = []
                data = json.loads(json_records)
                max_count = results[results.Total == results.Total.max()]
                json_records_max = max_count.reset_index().to_json(orient = 'records')
                max_data = []
                max_data = json.loads(json_records_max)
                min_count = results[results.Total == results.Total.min()]
                json_records_min = min_count.reset_index().to_json(orient = 'records')
                min_data = []
                min_data = json.loads(json_records_min)
                context = {'data':data,'present_columns':present_columns,'line_plot':line_plot,'max_data':max_data,'min_data':min_data,'insight':insight}
                df[insight] = df[insight].astype(str)
            else:
                insight_results = df.groupby(insight)[parameter].sum().reset_index()
                results = insight_results.rename(columns={insight_results.columns[1]:'Total'})
                present_columns = results.columns.to_list()
                json_records = results.reset_index().to_json(orient ='records')
                data = []
                data = json.loads(json_records)
                max_count = results[results.Total == results.Total.max()]
                json_records_max = max_count.reset_index().to_json(orient = 'records')
                max_data = []
                max_data = json.loads(json_records_max)
                min_count = results[results.Total == results.Total.min()]
                json_records_min = min_count.reset_index().to_json(orient = 'records')
                min_data = []
                min_data = json.loads(json_records_min)
                context = {'data': data,'present_columns':present_columns,'max_data':max_data,'min_data':min_data,'insight':insight}
                try:
                    results.plot.pie(y='Total',labeldistance=None,figsize=(9,9),autopct= lambda p:'{:.2f}% ({:,.0f})'.format(p,p*df[insight].count()/100))
                    labels = results[insight].tolist()
                    plt.legend(labels,title= 'LEGEND',bbox_to_anchor=(1,1))
                    plt.ylabel(None)
                    buf = BytesIO()
                    plt.savefig(buf, format='png',dpi = 300, bbox_inches='tight')
                    pie_chart = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
                    buf.close()
                    context.update({'pie_chart':pie_chart})
                except:
                    pass
                try:
                    results.plot.bar(x=insight,y='Total',figsize=(9,9))
                    addlabels(results[insight], results['Total'])
                    plt.ylabel('Count')
                    buf = BytesIO()
                    plt.savefig(buf, format='png',dpi = 300, bbox_inches='tight')
                    bar_chart = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
                    buf.close()
                    context.update({'bar_chart':bar_chart})
                except:
                    pass
            return render(request,'insightsapp/insight_results.html',context)



class MultiVarientInsights(View):
    def post(self,request):
        insight1 = request.POST.get('insight1')
        insight2 = request.POST.get('insight2')
        parameter = request.POST.get('parameter')
        if insight1 == insight2:
            request.session['err'] = 'Both the fields cannot be same !!!'
            return redirect('datainsights:insights')
        else:
            global df
            if parameter == 'Count':
                if insight1 in date_columns:
                    df[insight1] = pd.to_datetime(df[insight1])
                    insight_results = df.groupby([pd.Grouper(key=insight1, axis = 0, freq = 'M'),insight2]).count().reset_index()
                    insight_results = insight_results.iloc[:,0:3]
                    results = insight_results.rename(columns={insight_results.columns[2]:'Count'})
                    results[insight1] = results[insight1].dt.strftime("%Y-%b")
                    fig, ax = plt.subplots(figsize=(9,9))
                    sns.lineplot(data=results, x=insight1, y='Count', hue=insight2)
                    for i in ax.containers:
                        ax.bar_label(i)
                    plt.xlabel(insight1)
                    plt.ylabel('Count')
                    plt.xticks(rotation=90)
                    plt.legend(bbox_to_anchor=(1,1))
                    buf = BytesIO()
                    plt.savefig(buf, format='png',dpi = 300, bbox_inches='tight')
                    line_plot = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
                    buf.close()
                    present_columns = results.columns.to_list()
                    results.index=results.index.to_series().astype(str)
                    json_records = results.reset_index().to_json(orient ='records')
                    data = []
                    data = json.loads(json_records)
                    max_count = results[results.Count == results.Count.max()]
                    json_records_max = max_count.reset_index().to_json(orient = 'records')
                    max_data = []
                    max_data = json.loads(json_records_max)
                    min_count = results[results.Count == results.Count.min()]
                    json_records_min = min_count.reset_index().to_json(orient = 'records')
                    min_data = []
                    min_data = json.loads(json_records_min)
                    context = {'data':data,'present_columns':present_columns,'max_data':max_data,'min_data':min_data,'insight1':insight1,'insight2':insight2,'line_plot':line_plot}
                    df[insight1] = df[insight1].astype(str)
                else:
                    insight_results = df.groupby([insight1,insight2]).count().reset_index()
                    insight_results = insight_results.iloc[:,0:3]
                    results = insight_results.rename(columns={insight_results.columns[2]:'Count'})
                    fig, ax = plt.subplots(figsize=(9,9))
                    sns.set(font_scale=0.8)
                    sns.barplot(data=results, x=insight1, y='Count', hue=insight2)
                    for i in ax.containers:
                        ax.bar_label(i)
                    plt.xlabel(insight1)
                    plt.ylabel('Count')
                    plt.xticks(rotation=90)
                    plt.legend(bbox_to_anchor=(1,1))
                    buf = BytesIO()
                    plt.savefig(buf, format='png',dpi = 300, bbox_inches='tight')
                    bar_chart = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
                    buf.close()
                    present_columns = results.columns.to_list()
                    json_records = results.reset_index().to_json(orient ='records')
                    data = []
                    data = json.loads(json_records)
                    max_count = results[results.Count == results.Count.max()]
                    json_records_max = max_count.reset_index().to_json(orient = 'records')
                    max_data = []
                    max_data = json.loads(json_records_max)
                    min_count = results[results.Count == results.Count.min()]
                    json_records_min = min_count.reset_index().to_json(orient = 'records')
                    min_data = []
                    min_data = json.loads(json_records_min)
                    context = {'data': data,'present_columns':present_columns,'max_data':max_data,'min_data':min_data,'insight1':insight1,'insight2':insight2,'bar_chart':bar_chart}
            else:
                if insight1 in date_columns:
                    df[insight1] = pd.to_datetime(df[insight1])
                    insight_results = df.groupby([pd.Grouper(key=insight1, axis = 0, freq = 'M'),insight2])[parameter].sum().reset_index()
                    results = insight_results.rename(columns={insight_results.columns[2]:'Total'})
                    results[insight1] = results[insight1].dt.strftime("%Y-%b")
                    fig, ax = plt.subplots(figsize=(9,9))
                    sns.lineplot(data=results, x=insight1, y='Total', hue=insight2)
                    for i in ax.containers:
                        ax.bar_label(i)
                    plt.xlabel(insight1)
                    plt.ylabel('Total')
                    plt.xticks(rotation=90)
                    plt.legend(bbox_to_anchor=(1,1))
                    buf = BytesIO()
                    plt.savefig(buf, format='png',dpi = 300, bbox_inches='tight')
                    line_plot = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
                    buf.close()
                    present_columns = results.columns.to_list()
                    results.index=results.index.to_series().astype(str)
                    json_records = results.reset_index().to_json(orient ='records')
                    data = []
                    data = json.loads(json_records)
                    max_count = results[results.Total == results.Total.max()]
                    json_records_max = max_count.reset_index().to_json(orient = 'records')
                    max_data = []
                    max_data = json.loads(json_records_max)
                    min_count = results[results.Total == results.Total.min()]
                    json_records_min = min_count.reset_index().to_json(orient = 'records')
                    min_data = []
                    min_data = json.loads(json_records_min)
                    context = {'data':data,'present_columns':present_columns,'max_data':max_data,'min_data':min_data,'insight1':insight1,'insight2':insight2,'line_plot':line_plot}
                    df[insight1] = df[insight1].astype(str)
                else:
                    insight_results = df.groupby([insight1,insight2])[parameter].sum().reset_index()
                    results = insight_results.rename(columns={insight_results.columns[2]:'Total'})
                    fig, ax = plt.subplots(figsize=(9,9))
                    sns.set(font_scale=0.8)
                    sns.barplot(data=results, x=insight1, y='Total', hue=insight2)
                    for i in ax.containers:
                        ax.bar_label(i)
                    plt.xlabel(insight1)
                    plt.ylabel('Total')
                    plt.xticks(rotation=90)
                    plt.legend(bbox_to_anchor=(1,1))
                    buf = BytesIO()
                    plt.savefig(buf, format='png',dpi = 300, bbox_inches='tight')
                    bar_chart = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
                    buf.close()
                    present_columns = results.columns.to_list()
                    json_records = results.reset_index().to_json(orient ='records')
                    data = []
                    data = json.loads(json_records)
                    max_count = results[results.Total == results.Total.max()]
                    json_records_max = max_count.reset_index().to_json(orient = 'records')
                    max_data = []
                    max_data = json.loads(json_records_max)
                    min_count = results[results.Total == results.Total.min()]
                    json_records_min = min_count.reset_index().to_json(orient = 'records')
                    min_data = []
                    min_data = json.loads(json_records_min)
                    context = {'data': data,'present_columns':present_columns,'max_data':max_data,'min_data':min_data,'insight1':insight1,'insight2':insight2,'bar_chart':bar_chart}

            return render(request,'insightsapp/multivarientinsight_results.html',context)
