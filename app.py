import streamlit as st
import functions
import matplotlib.pyplot as plt

st.sidebar.title('Whatsapp Chat Analyzer')

## uploading a file option
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode("utf-8")
        df = functions.fetch_data(data)

        
        # fetch unique users
        user_list = df['User'].unique().tolist()
        user_list.sort()
        user_list.insert(0,"Overall")

        selected_user = st.sidebar.selectbox("Show analysis wrt",user_list)

        if st.sidebar.button("Show Analysis"):

        # Stats Area
            num_messages,words, num_media_messages, num_links = functions.fetch_stats(selected_user,df)
            st.title("Analysis Of Chat")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.subheader("Messages")
                st.title(num_messages)
            with col2:
                st.subheader("Total Words")
                st.title(words)
            with col3:
                st.subheader("Media File")
                st.title(num_media_messages)
            with col4:
                st.subheader("Links Shared")
                st.title(num_links)
            # monthly timeline
            st.title("Monthly Timeline")
            timeline = functions.monthly_timeline(selected_user,df)
            fig,ax = plt.subplots()
            ax.plot(timeline['Time'], timeline['Message'],color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # daily timeline
            st.title("Daily Timeline")
            daily_timeline = functions.daily_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['date'], daily_timeline['Message'], color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # Analysis of time when group was most active

            st.title('Analysis of time when group/selectd user was most active')
            time_analysis=functions.analysis_time(selected_user,df)
            fig, ax = plt.subplots()
            ax.bar(time_analysis.index, time_analysis.values, color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)



            # finding the busiest users in the group(Group level)
            if selected_user == 'Overall':
                st.title('Most Active Users')
                x,new_df = functions.most_active_users(df)
                fig, ax = plt.subplots()

                col1, col2 = st.columns(2)

                with col1:
                    ax.bar(x.index, x.values,color='green')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col2:
                    st.dataframe(new_df)
            
            # finding the Least active users in the group(Group level)
            if selected_user == 'Overall':
                st.title('Least Active Users')
                x,new_df = functions.least_active_users(df)
                fig, ax = plt.subplots()

                col1, col2 = st.columns(2)

                with col1:
                    ax.bar(x.index, x.values,color='red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col2:
                    st.dataframe(new_df)
            # activity map
            st.title('Activity Map')
            col1,col2,col3 = st.columns(3)

            with col1:
                st.subheader("Most busy day")
                busy_day = functions.week_activity_map(selected_user,df)
                fig,ax = plt.subplots()
                ax.bar(busy_day.index,busy_day.values,color='purple')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.subheader("Most busy month")
                busy_month = functions.month_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values,color='orange')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            
            ## Most Busy Year
            with col3:
                st.subheader("Most Busy Year")
                busy_year = functions.year_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_year.index, busy_year.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            
            # WordCloud
            st.title("Wordcloud")
            df_wc = functions.create_wordcloud(selected_user,df)
            fig,ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)

            # most common words
            most_common_df = functions.most_common_words(selected_user,df)

            fig,ax = plt.subplots()

            ax.barh(most_common_df[0],most_common_df[1])
            plt.xticks(rotation='vertical')

            st.title('Most commmon words')
            st.pyplot(fig)

            # emoji analysis
            emoji_df = functions.emoji_helper(selected_user,df)
            st.title("Emoji Analysis")
            if emoji_df.shape[0]>0:

                col1,col2 = st.columns(2)

                with col1:
                    st.dataframe(emoji_df)
                with col2:
                    fig,ax = plt.subplots()
                    ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.2f")
                    st.pyplot(fig)
            else:
                st.subheader('No emojis shared')
            
            # Sentiment Analysis
            st.title('Sentiment Analysis')
            a=sum(df["positive"])
            b=sum(df["negative"])
            c=sum(df["neutral"])
            sentiment =functions.score(a,b,c)
            st.subheader(f'Sentiment of the group is {sentiment}')

            ## Individual Sentiment Analysis
            if selected_user != 'Overall':
                st.title('Sentiment of Selected User')
                x=sum(df[df['User']==selected_user]['positive'])
                y=sum(df[df['User']==selected_user]['negative'])
                z=sum(df[df['User']==selected_user]['neutral'])
                sentiment=functions.score(x,y,z)
                st.subheader(f'Sentiment of {selected_user} is {sentiment}')
            
            ## All users sentiment analysis
            st.title('Sentiment Analysis of all group members')
            df_new=df[['User','positive','negative','neutral']]
            st.dataframe(df_new)






        

        