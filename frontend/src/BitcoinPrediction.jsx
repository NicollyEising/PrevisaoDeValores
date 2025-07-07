import React, { useEffect, useState } from "react";
import axios from "axios";
import {
  Container,
  Header,
  Loader,
  Message,
  Icon,
  Grid,
  Segment,
  Card,
  Statistic,
} from "semantic-ui-react";
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";
import "semantic-ui-css/semantic.min.css";

function BitcoinPrediction() {
  const [prediction, setPrediction] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const [predRes, chartRes] = await Promise.all([
          axios.get("http://127.0.0.1:8000/predict"),
          axios.get("http://127.0.0.1:8000/chart-data"),
        ]);
        setPrediction(predRes.data);

        const combined = chartRes.data.dates.map((date, i) => ({
          date,
          Real: chartRes.data.real_close[i],
          LSTM: chartRes.data.lstm_close[i],
          XGBoost: chartRes.data.xgb_close[i],
          Ensemble: chartRes.data.ensemble_close[i],
        }));
        setChartData(combined);
      } catch {
        setError("Erro ao carregar dados da API");
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading) {
    return (
      <Container textAlign="center" style={{ marginTop: "4em" }}>
        <Loader active inline="centered" size="large" content="Carregando previsão..." />
      </Container>
    );
  }

  if (error) {
    return (
      <Container textAlign="center" style={{ marginTop: "4em" }}>
        <Message negative icon>
          <Icon name="exclamation triangle" />
          <Message.Content>
            <Message.Header>Erro</Message.Header>
            {error}
          </Message.Content>
        </Message>
      </Container>
    );
  }

  return (
    <Container style={{ padding: "2em" }}>
      <Header as="h2" dividing>
        <Icon name="chart line" />
        <Header.Content>Previsão do Preço do Bitcoin</Header.Content>
      </Header>

      <Grid stackable columns={3} style={{ marginBottom: "2em" }}>
        <Grid.Column>
          <Card color="green" fluid>
            <Card.Content>
              <Card.Header textAlign="center">Fechamento (Close)</Card.Header>
              <Card.Description textAlign="center">
                <Statistic size="tiny">
                  <Statistic.Value>${prediction.close.toFixed(2)}</Statistic.Value>
                  <Statistic.Label>MAPE: {prediction.mape_close.toFixed(2)}%</Statistic.Label>
                </Statistic>
              </Card.Description>
            </Card.Content>
          </Card>
        </Grid.Column>
        <Grid.Column>
          <Card color="red" fluid>
            <Card.Content>
              <Card.Header textAlign="center">Baixa (Low)</Card.Header>
              <Card.Description textAlign="center">
                <Statistic size="tiny">
                  <Statistic.Value>${prediction.low.toFixed(2)}</Statistic.Value>
                  <Statistic.Label>MAPE: {prediction.mape_low.toFixed(2)}%</Statistic.Label>
                </Statistic>
              </Card.Description>
            </Card.Content>
          </Card>
        </Grid.Column>
        <Grid.Column>
          <Card color="blue" fluid>
            <Card.Content>
              <Card.Header textAlign="center">Alta (High)</Card.Header>
              <Card.Description textAlign="center">
                <Statistic size="tiny">
                  <Statistic.Value>${prediction.high.toFixed(2)}</Statistic.Value>
                  <Statistic.Label>MAPE: {prediction.mape_high.toFixed(2)}%</Statistic.Label>
                </Statistic>
              </Card.Description>
            </Card.Content>
          </Card>
        </Grid.Column>
      </Grid>

      <Segment>
        <Header as="h3">
          <Icon name="area chart" />
          <Header.Content>Gráfico de Preço de Fechamento</Header.Content>
        </Header>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={chartData}>
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Legend verticalAlign="top" height={36} />
            <Line type="monotone" dataKey="Real" stroke="#000" dot={false} />
            <Line type="monotone" dataKey="LSTM" stroke="#2ca02c" dot={false} />
            <Line type="monotone" dataKey="XGBoost" stroke="#ff7f0e" dot={false} />
            <Line type="monotone" dataKey="Ensemble" stroke="#1f77b4" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </Segment>
    </Container>
  );
}

export default BitcoinPrediction;
