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
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from dateutil.relativedelta import relativedelta
from django.http import HttpResponse
from django.template.loader import render_to_string
import random

# Create your views here.

df = pd.DataFrame()
suitable_columns =[]
date_columns =[]
numeric_columns =[]

def get_color():
    hexadecimal = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])]
    return hexadecimal


class Home(View):
    def get(self,request):
        return render(request,'insightsapp/home.html')
    def post(self,request):
        file = request.FILES["file"]
        extension = os.path.splitext(file.name)[1]
        if extension == '.xlsx':    
            df1 = pd.read_excel(file)
        elif extension=='.csv' or '.txt':
            df1 = pd.read_csv(file, encoding = 'unicode_escape')
        else:
            html = render_to_string('insightsapp/home.html',{'err':'File must be excel worksheet, csv or txt'})
            return HttpResponse(html)
        global df
        df1.columns = df1.columns.str.replace(' ','_')
        df = df1.copy()
        present_columns = df.columns.to_list()
        for i in present_columns:
            if df[i].dtypes == 'datetime64[ns]':
                df[i] = df[i].astype(str)
        json_records = df.head(15).reset_index().to_json(orient ='records')
        data = []
        data = json.loads(json_records)
        context = {'data': data,'present_columns':present_columns}
        html = render_to_string('insightsapp/home.html',context)
        return HttpResponse(html)



def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i], ha = 'center')


insight = None
parameter = None
def forecast_parameters(a,b):
    global insight,parameter
    insight = a
    parameter = b


class SingleInsights(View):
    def get(self,request):
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

        context = {'suitable_columns':suitable_columns,'date_columns':date_columns,'numeric_columns':numeric_columns}
        return render(request, 'insightsapp/insights.html', context)

    def post(self,request):
        insight = request.POST.get('insight')
        parameter = request.POST.get('parameter')
        forecast_parameters(insight,parameter)
        insight_type =request.POST.get('insight_type')
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
                context = {'data':data,'present_columns':present_columns,'line_plot':line_plot,'max_data':max_data,'min_data':min_data,'insight':insight,'parameter':parameter,'pred':'possible'}
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
                context = {'data': data,'present_columns':present_columns,'max_data':max_data,'min_data':min_data,'insight':insight,'parameter':parameter}
                try:
                    labels = results[insight].tolist()
                    color = []
                    for i in labels:
                        color = color+get_color()
                    results.plot.pie(y='Count',labeldistance=None,figsize=(15,15),colors=color)
                    percents = results['Count'].to_numpy() * 100 / results['Count'].to_numpy().sum()
                    amt = results['Count'].to_numpy()
                    plt.legend(labels,title= 'LEGEND',labels=['%s - %.0f (%1.1f %%)' % (l, s, t) for l, s, t in zip(results[insight],amt,percents)],bbox_to_anchor=(1,1),fontsize=20)
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
                if results.shape[0] > 10:
                    results = results.sort_values(by=['Count'],ascending=False)
                    top = results.head(10).reset_index()
                    top.plot.bar(x=insight,y='Count',figsize=(9,9))
                    addlabels(top[insight],top['Count'])
                    plt.title('Bar chart for top 10 values')
                    plt.ylabel('Count')
                    buf = BytesIO()
                    plt.savefig(buf, format='png',dpi = 300, bbox_inches='tight')
                    bar_top = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
                    buf.close()
                    context.update({'top':top,'bar_top':bar_top})
            return render(request,'insightsapp/insight_results.html',context)
        else:
            if insight_type == 'Sum':
                if insight in date_columns:
                    df[insight] = pd.to_datetime(df[insight])
                    insight_results = df.resample('M',on = insight)[parameter].sum()
                    insight_results = insight_results.to_frame()
                    results = insight_results.rename(columns={insight_results.columns[0]:'Total'})
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
                    context = {'data':data,'present_columns':present_columns,'line_plot':line_plot,'max_data':max_data,'min_data':min_data,'insight':insight,'parameter':parameter,'pred':'possible'}
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
                    context = {'data': data,'present_columns':present_columns,'max_data':max_data,'min_data':min_data,'insight':insight,'parameter':parameter}
                    try:
                        labels = results[insight].tolist()
                        color = []
                        for i in labels:
                            color = color+get_color()
                        results.plot.pie(y='Total',labeldistance=None,figsize=(15,15),colors=color)
                        labels = results[insight].tolist()
                        percents = results['Total'].to_numpy() * 100 / results['Total'].to_numpy().sum()
                        amt = results['Total'].to_numpy()
                        plt.legend(labels,title= 'LEGEND',labels=['%s - %.0f (%1.1f %%)' % (l, s, t) for l, s, t in zip(results[insight],amt,percents)],bbox_to_anchor=(1,1),fontsize=20)
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
                        count = results['Total'].to_list()
                        context.update({'bar_chart':bar_chart})
                    except:
                        pass
                    if results.shape[0] > 10:
                        results = results.sort_values(by=['Total'],ascending = False)
                        top = results.head(10).reset_index()
                        top.plot.bar(x=insight,y='Total',figsize=(9,9))
                        addlabels(top[insight],top['Total'])
                        plt.title('Bar chart for top 10 values')
                        plt.ylabel('Total')
                        buf = BytesIO()
                        plt.savefig(buf, format='png',dpi = 300, bbox_inches='tight')
                        bar_top = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
                        buf.close()
                        context.update({'top':top,'bar_top':bar_top})
            if insight_type == 'Average':
                if insight in date_columns:
                    df[insight] = pd.to_datetime(df[insight])
                    insight_results = df.resample('M',on = insight)[parameter].mean()
                    insight_results = insight_results.to_frame()
                    results = insight_results.rename(columns={insight_results.columns[0]:'Total'})
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
                    context = {'data':data,'present_columns':present_columns,'line_plot':line_plot,'max_data':max_data,'min_data':min_data,'insight':insight,'parameter':parameter,'pred':'possible'}
                    df[insight] = df[insight].astype(str)
                else:
                    insight_results = df.groupby(insight)[parameter].mean().reset_index()
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
                    context = {'data': data,'present_columns':present_columns,'max_data':max_data,'min_data':min_data,'insight':insight,'parameter':parameter}
                    try:
                        labels = results[insight].tolist()
                        color = []
                        for i in labels:
                            color = color+get_color()
                        results.plot.pie(y='Total',labeldistance=None,figsize=(15,15),colors=color)
                        labels = results[insight].tolist()
                        percents = results['Total'].to_numpy() * 100 / results['Total'].to_numpy().sum()
                        amt = results['Total'].to_numpy()
                        plt.legend(labels,title= 'LEGEND',labels=['%s - %.0f (%1.1f %%)' % (l, s, t) for l, s, t in zip(results[insight],amt,percents)],bbox_to_anchor=(1,1),fontsize=20)
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
                        count = results['Total'].to_list()
                        context.update({'bar_chart':bar_chart})
                    except:
                        pass
                    if results.shape[0] > 10:
                        results = results.sort_values(by=['Total'],ascending = False)
                        top = results.head(10).reset_index()
                        top.plot.bar(x=insight,y='Total',figsize=(9,9))
                        addlabels(top[insight],top['Total'])
                        plt.title('Bar chart for top 10 values')
                        plt.ylabel('Total')
                        buf = BytesIO()
                        plt.savefig(buf, format='png',dpi = 300, bbox_inches='tight')
                        bar_top = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
                        buf.close()
                        context.update({'top':top,'bar_top':bar_top})
            context.update({'insight_type':insight_type})
            return render(request,'insightsapp/insight_results.html',context)



class MultiVarientInsights(View):
    def post(self,request):
        insight1 = request.POST.get('insight1')
        insight2 = request.POST.get('insight2')
        parameter = request.POST.get('parameter')
        insight_type = request.POST.get('insight_type')
        global df
        if parameter == 'Count':
            if insight1 in date_columns:
                df[insight1] = pd.to_datetime(df[insight1])
                insight_results = df.groupby([pd.Grouper(key=insight1, axis = 0, freq = 'M'),insight2]).count().reset_index()
                insight_results = insight_results.iloc[:,0:3]
                results = insight_results.rename(columns={insight_results.columns[2]:'Count'})
                total_count = results.groupby(insight1)['Count'].sum().to_list()
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
                context = {'data':data,'present_columns':present_columns,'max_data':max_data,'min_data':min_data,'insight1':insight1,'insight2':insight2,'parameter':parameter,'line_plot':line_plot,'total_for_groups':total_count}
                df[insight1] = df[insight1].astype(str)
            else:
                insight_results = df.groupby([insight1,insight2]).count().reset_index()
                insight_results = insight_results.iloc[:,0:3]
                results = insight_results.rename(columns={insight_results.columns[2]:'Count'})
                total_count = results.groupby(insight1)['Count'].sum().to_list()
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
                print(results)
                context = {'data': data,'present_columns':present_columns,'max_data':max_data,'min_data':min_data,'insight1':insight1,'insight2':insight2,'parameter':parameter,'bar_chart':bar_chart,'total_for_groups':total_count}
        else:
            if insight_type == 'Sum':
                if insight1 in date_columns:
                    df[insight1] = pd.to_datetime(df[insight1])
                    insight_results = df.groupby([pd.Grouper(key=insight1, axis = 0, freq = 'M'),insight2])[parameter].sum().reset_index()
                    results = insight_results.rename(columns={insight_results.columns[2]:'Total'})
                    total_by_groups = results.groupby(insight1)['Total'].sum().to_list()
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
                    context = {'data':data,'present_columns':present_columns,'max_data':max_data,'min_data':min_data,'insight1':insight1,'insight2':insight2,'parameter':parameter,'line_plot':line_plot,'total_for_groups':total_by_groups}
                    df[insight1] = df[insight1].astype(str)
                else:
                    insight_results = df.groupby([insight1,insight2])[parameter].sum().reset_index()
                    results = insight_results.rename(columns={insight_results.columns[2]:'Total'})
                    total_by_groups = results.groupby(insight1)['Total'].sum().to_list()
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
                    context = {'data': data,'present_columns':present_columns,'max_data':max_data,'min_data':min_data,'insight1':insight1,'insight2':insight2,'parameter':parameter,'bar_chart':bar_chart,'total_for_groups':total_by_groups}
            if insight_type == 'Average':
                if insight1 in date_columns:
                    df[insight1] = pd.to_datetime(df[insight1])
                    insight_results = df.groupby([pd.Grouper(key=insight1, axis = 0, freq = 'M'),insight2])[parameter].mean().reset_index()
                    results = insight_results.rename(columns={insight_results.columns[2]:'Total'})
                    total_by_groups = results.groupby(insight1)['Total'].mean().to_list()
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
                    context = {'data':data,'present_columns':present_columns,'max_data':max_data,'min_data':min_data,'insight1':insight1,'insight2':insight2,'parameter':parameter,'line_plot':line_plot,'total_for_groups':total_by_groups}
                    df[insight1] = df[insight1].astype(str)
                else:
                    insight_results = df.groupby([insight1,insight2])[parameter].mean().reset_index()
                    results = insight_results.rename(columns={insight_results.columns[2]:'Total'})
                    total_by_groups = results.groupby(insight1)['Total'].mean().to_list()
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
                    context = {'data': data,'present_columns':present_columns,'max_data':max_data,'min_data':min_data,'insight1':insight1,'insight2':insight2,'parameter':parameter,'bar_chart':bar_chart,'total_for_groups':total_by_groups}
        context.update({'insight_type':insight_type})
        return render(request,'insightsapp/multivarientinsight_results.html',context)



class Forecast(View):
    def get(self,request):
        return render(request,'insightsapp/forecast.html')
    def post(self,request):
        global insight,parameter
        no = int(request.POST.get('no'))
        if parameter == 'Count':
            df[insight] = pd.to_datetime(df[insight])
            insight_results = df.resample('M',on = insight).count()
            insight_results = insight_results.iloc[:,0:1]
            results = insight_results.rename(columns={insight_results.columns[0]:'Count'})
            results = results.to_period('M').reset_index()
            y = results['Count']
            y = y.values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler = scaler.fit(y)
            y = scaler.transform(y)
            n_lookback = results.shape[0]-(no+1)  # length of input sequences (lookback period)
            n_forecast = no # length of output sequences (forecast period)
            X = []
            Y = []
            for i in range(n_lookback, len(y) - n_forecast + 1):
                X.append(y[i - n_lookback: i])
                Y.append(y[i: i + n_forecast])
            X = np.array(X)
            Y = np.array(Y)
            try:
                model = Sequential()
                model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
                model.add(LSTM(units=50))
                model.add(Dense(n_forecast))
                model.compile(loss='mean_squared_error', optimizer='adam')
                model.fit(X, Y, epochs=100, batch_size=32, verbose=0)
                X_ = y[- n_lookback:]  # last available input sequence
                X_ = X_.reshape(1, n_lookback, 1)
                Y_ = model.predict(X_).reshape(-1, 1)
                Y_ = scaler.inverse_transform(Y_)

                df_past = results
                df_past.rename(columns={insight: 'Date', 'Count': 'Actual'}, inplace=True)
                df_past['Date'] = df_past['Date'].dt.to_timestamp('s')
                df_past['Date'] = pd.to_datetime(df_past['Date'])
                df_past['Forecast'] = np.nan
                df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]
    
                df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
                df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + relativedelta(months=1), periods=n_forecast,freq='MS')
                df_future['Forecast'] = Y_.flatten()
                df_future['Actual'] = np.nan
                df_future['Forecast']=df_future['Forecast'].apply(np.floor)
                df_future['Forecast'] = df_future['Forecast'].astype(int)
                forecast_results = df_past.append(df_future).set_index('Date')
                df_future = df_future.drop('Actual',axis=1)
                df_future['Date'] = df_future['Date'].dt.strftime("%Y-%b")
                df_future['Date'] = df_future['Date'].astype(str)
                present_columns = df_future.columns.to_list()
                json_records = df_future.reset_index().to_json(orient ='records')
                data = []
                data = json.loads(json_records)
                forecast_results.plot()
                buf = BytesIO()
                plt.savefig(buf, format='png',dpi = 300, bbox_inches='tight')
                line_plot = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
                buf.close()
                context={'data':data,'present_columns':present_columns,'line_plot':line_plot}
            except:
                err = 'Not enough data for '+ str(no) +' predictions. Try with a smaller number.'
                context={'err':err}
        else:
            df[insight] = pd.to_datetime(df[insight])
            insight_results = df.resample('M',on = insight)[parameter].sum()
            insight_results = insight_results.to_frame()
            results = insight_results.rename(columns={insight_results.columns[0]:'Total'})
            results = results.to_period('M').reset_index()
            y = results['Total']
            y = y.values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler = scaler.fit(y)
            y = scaler.transform(y)
            n_lookback = results.shape[0]-(no+1)  # length of input sequences (lookback period)
            n_forecast = no # length of output sequences (forecast period)
            X = []
            Y = []
            for i in range(n_lookback, len(y) - n_forecast + 1):
                X.append(y[i - n_lookback: i])
                Y.append(y[i: i + n_forecast])
            X = np.array(X)
            Y = np.array(Y)
            try:
                model = Sequential()
                model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
                model.add(LSTM(units=50))
                model.add(Dense(n_forecast))
                model.compile(loss='mean_squared_error', optimizer='adam')
                model.fit(X, Y, epochs=100, batch_size=32, verbose=0)
                X_ = y[- n_lookback:]  # last available input sequence
                X_ = X_.reshape(1, n_lookback, 1)
                Y_ = model.predict(X_).reshape(-1, 1)
                Y_ = scaler.inverse_transform(Y_)

                df_past = results
                df_past.rename(columns={insight: 'Date', 'Total': 'Actual'}, inplace=True)
                df_past['Date'] = df_past['Date'].dt.to_timestamp('s')
                df_past['Date'] = pd.to_datetime(df_past['Date'])
                df_past['Forecast'] = np.nan
                df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]
    
                df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
                df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + relativedelta(months=1), periods=n_forecast,freq='MS')
                df_future['Forecast'] = Y_.flatten()
                df_future['Actual'] = np.nan
                df_future['Forecast'] = df_future['Forecast'].round(2)
                df_past['Date']=df_past['Date'].dt.to_period('M')
                df_future['Date']=df_future['Date'].dt.to_period('M')
                forecast_results = df_past.append(df_future).set_index('Date')
                df_future = df_future.drop('Actual',axis=1)
                df_future['Date'] = df_future['Date'].dt.strftime("%Y-%b")
                df_future['Date'] = df_future['Date'].astype(str)
                present_columns = df_future.columns.to_list()
                
                json_records = df_future.reset_index().to_json(orient ='records')
                data = []
                data = json.loads(json_records)
                forecast_results.plot()
                buf = BytesIO()
                plt.savefig(buf, format='png',dpi = 300, bbox_inches='tight')
                line_plot = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
                buf.close()
                context={'data':data,'present_columns':present_columns,'line_plot':line_plot}
            except:
                err ='No enough data for '+ str(no) +' predictions. Try with a smaller number.'
                context ={'err':err}
        context.update({'insight':insight,'parameter':parameter})
        # template = loader.get_template(template_name='insightsapp/forecast.html')
        # result = template.render(context,request) 
        # return HttpResponse(result)
        html = render_to_string('insightsapp/forecast.html',context)
        return HttpResponse(html)
